from flask import Flask, abort, request, jsonify , Response
from tempfile import NamedTemporaryFile


AUDIO_RECOG = True
FACE_RECOG = True
TRANSCRIBE = True

if TRANSCRIBE:
    import whisper
    import torch    
    whispher_model = whisper.load_model("medium.en")

if AUDIO_RECOG:
    from pyannote.audio import Model
    from pyannote.audio import Inference
    from scipy.spatial.distance import cdist
    import os

    audio_recog_model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
    inference = Inference(audio_recog_model, window="whole")

    stored_Audio_embeddings = {}

    folder_path= "persons"
    for person_folder in os.listdir(folder_path):
        person_folder_path = os.path.join(folder_path, person_folder)
        if not os.path.isdir(person_folder_path):
            continue  # Skip if it's not a folder

        # Path to the face image file
        audio_folder_path = os.path.join(person_folder_path, "audio")
        sample_audio_path = os.path.join(audio_folder_path, "sample.wav")

        if os.path.isfile(sample_audio_path):
            cal_audio_embedding = inference(sample_audio_path)
            cal_audio_embedding = cal_audio_embedding.reshape(1,512)
            stored_Audio_embeddings[person_folder] = cal_audio_embedding

if FACE_RECOG:
    
    import cv2
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image
    import os
    import numpy as np
    import time

    face_recog_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_recog_app.prepare(ctx_id=0, det_size=(640, 640))

    handler = insightface.model_zoo.get_model('buffalo_l')
    handler.prepare(ctx_id=0)

    face_stored_embeddings = {}

    folder_path= "persons"
    for person_folder in os.listdir(folder_path):
        person_folder_path = os.path.join(folder_path, person_folder)
        if not os.path.isdir(person_folder_path):
            continue  # Skip if it's not a folder

        # Path to the face image file
        face_folder_path = os.path.join(person_folder_path, "face")
        face_image_path = os.path.join(face_folder_path, "img.jpg")

        if os.path.isfile(face_image_path):
            # Read the image using OpenCV
            image = cv2.imread(face_image_path)
            #img_crop = mtcnn(image)
            faces = face_recog_app.get(image)
            embeddings = handler.get(image, faces[0]) #resnet(img_crop.unsqueeze(0))
            face_stored_embeddings[person_folder] = embeddings

app = Flask(__name__)

if TRANSCRIBE:
    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        if not request.files:
            abort(400)
        results = []
        for filename, handle in request.files.items():
            temp = NamedTemporaryFile()
            handle.save(temp)
            result = whispher_model.transcribe(temp.name)
            results.append(result['text'] ) 
        return jsonify({'results': results})

if AUDIO_RECOG:
    @app.route('/audio_recog', methods=['POST'])
    def audio_recog():
        if not request.files:
            abort(400)
        results = []
        for filename, handle in request.files.items():
            temp = NamedTemporaryFile()
            handle.save(temp)
            
            curr_embedding = inference(temp.name)
            curr_embedding = curr_embedding.reshape(1,512)
            
            sim = 1
            detected_p = None
            for name, emb in stored_Audio_embeddings.items():
                distance = cdist(emb, curr_embedding, metric="cosine")
                if distance < sim:
                    sim = distance
                    detected_p = name
            out = {"Detected": detected_p , "Sim": float(sim)}
            print(out)
            results.append( [out] ) 
        return jsonify(out)

if FACE_RECOG:
    @app.route('/face_recog', methods=['POST'])
    def face_recog():
        if not request.files:
            abort(400)

        # Handle the image data
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        faces = face_recog_app.get(image)
        if len(faces) != 0:
            for i in range(len(faces)):
                det_embd = handler.get(image, faces[i])
                detected_p = None
                accu = -1
                for key, value in face_stored_embeddings.items():
                    recog = handler.compute_sim(value, det_embd )
                    if recog > accu:
                        detected_p = key
                        accu = recog
                color = (0,255,0)
                if accu < 0.2:
                    detected_p = None
                    accu = 0
                    color = (0,255,0)
                    #image = app.draw_on(image, [faces[i]] )
                if detected_p is not None:
                    image = face_recog_app.draw_on_mod(image, [faces[i]], detected_p , str(round(accu , 2)) , color)

        
        cv2.imwrite('processed_image.jpg', image)
        _, image_encoded = cv2.imencode('.jpg', image)
        
        # for filename, handle in request.files.items():
        #     temp = NamedTemporaryFile()
        #     handle.save(temp)

        #     faces = app.get(image)

        #     print(faces)

        response = Response(image_encoded.tobytes(), content_type='image/jpeg')
    

        return response #jsonify({'results': image_bytes })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5001)
