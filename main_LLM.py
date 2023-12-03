from flask import Flask, abort, request, jsonify , Response
from tempfile import NamedTemporaryFile
from helper_chatgpt import gptResponse
import time

AUDIO_RECOG = True
FACE_RECOG = True
TRANSCRIBE = True
EMOTION = True
WAKE_WORD = True

from LLM_code.LLM1 import LLMResponse

if EMOTION:
        
    # import libraries
    import cv2
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from torchvision import transforms
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = 'cuda' if use_cuda else 'cpu'
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import timm

    PATH='models/emotions.pt'
    try:
        model = torch.load(PATH,map_location=torch.device('cpu'))
        model=model.to(device)
        model.eval()
        
    except Exception as e:
        print("Error loading the model:", e)
        import traceback
        traceback.print_exc()

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
            stored_Audio_embeddings[person_folder.lower()] = cal_audio_embedding

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
    @app.route('/audio_identify', methods=['POST'])
    def audio_identify():
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
    
    @app.route('/audio_recog', methods=['POST'])
    def audio_recog():
        user = str( request.form.get('user', 'None'))
        print(user)
        handle =  request.files['audio']
        temp = NamedTemporaryFile()
        handle.save(temp)
        
        # results = []

        curr_embedding = inference(temp.name)
        curr_embedding = curr_embedding.reshape(1,512)

        distance = cdist(stored_Audio_embeddings[user.lower()], curr_embedding, metric="cosine")
        out = {"Sim": float(distance) }
        # results.append( [out] ) 

        return jsonify(out)
    
    @app.route('/give_embd', methods=['POST'])
    def give_embd():
        
        handle =  request.files['audio1']
        temp = NamedTemporaryFile()
        handle.save(temp)
        embedding1 = inference(temp.name)
        embedding1 = embedding1.reshape(1,512)
        
        handle =  request.files['audio2']
        temp = NamedTemporaryFile()
        handle.save(temp)
        embedding2 = inference(temp.name)
        embedding2 = embedding2.reshape(1,512)
        
        distance = 1 - cdist(embedding1, embedding2, metric="cosine")
        out = {"Sim": float(distance) }
        
        return jsonify(out)
    
    

def give_emotion(face, img):
    (x1,y1) = int(face['bbox'][0]), int(face['bbox'][1])
    (x2,y2) = int(face['bbox'][2]) , int(face['bbox'][3]) 

    face_img = img[y1:y2,x1:x2,:]
            
    idx_to_class={0: 'Sad', 1: 'Neutral', 2: 'Neutral', 3: 'Sad', 4: 'Happy', 5: 'Neutral', 6: 'Sad', 7: 'Neutral'}
    IMG_SIZE=224
    test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ]
    )
    img_tensor = test_transforms(Image.fromarray(face_img))
    img_tensor.unsqueeze_(0)

    scores = model(img_tensor.to(device))
    scores=scores[0].data.cpu().numpy()
    label = idx_to_class[np.argmax(scores)]

    return label

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
                label = None
                if EMOTION:
                    label = give_emotion( faces[i] , image)

                
                image = face_recog_app.draw_on_mod(image, [faces[i]], detected_p , label , str(round(accu , 2)) , color)

        
        cv2.imwrite('processed_image.jpg', image)
        _, image_encoded = cv2.imencode('.jpg', image)
        
        # for filename, handle in request.files.items():
        #     temp = NamedTemporaryFile()
        #     handle.save(temp)

        #     faces = app.get(image)

        #     print(faces)

        response = Response(image_encoded.tobytes(), content_type='image/jpeg')
        return response
        
        
    @app.route('/face_sim', methods=['POST'])
    def face_sim():
        if not request.files:
            abort(400)

        # Handle the image data
        image_file = request.files['image1']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        faces = face_recog_app.get(image)
        if len(faces) != 0:
            for i in range(len(faces)):
                det_embd1 = handler.get(image, faces[i])

        # Handle the image data
        image_file = request.files['image2']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        faces = face_recog_app.get(image)
        if len(faces) != 0:
            for i in range(len(faces)):
                det_embd2 = handler.get(image, faces[i])
                
        recog = handler.compute_sim(det_embd1, det_embd2 )
        
        out = {"Sim": float(recog) }
        
        return jsonify(out)


if WAKE_WORD:
    import pvporcupine
    import os
    import json
    porc_model_path_ppn = "models/hello-kai_en_linux_v2_2_0.ppn"
    #pico_key = os.environ["PICOVOICE_API_KEY"]
    porcupine = pvporcupine.create(access_key="mIyfVBxZvAvw4wEhVoXOBagmIy1f+EViskPNpAgzhZy8KkPQYIGzhA==", keyword_paths=[porc_model_path_ppn])

if WAKE_WORD:
    @app.route('/wake_word', methods=['POST'])
    def wakeword_recog():
        int_list = json.loads(request.files['audio'].read().decode('utf-8'))
        keyword_index = porcupine.process(int_list)
        return jsonify(keyword_index)

@app.route('/complete', methods=['POST'])
def complete():
    user = str( request.form.get('user', 'None'))
    audio_auth = str(request.form.get('audio_auth', False))
    response_method = str(request.form.get('response_method', "chatGPT"))

    handle =  request.files['audio']
    temp = NamedTemporaryFile()
    handle.save(temp)

    distance = 0.0
    time_ar = None

    if audio_auth == "True":
        print("audio auth")
        start_time = time.time()
        curr_embedding = inference(temp.name)
        curr_embedding = curr_embedding.reshape(1,512)
        distance = cdist(stored_Audio_embeddings[user.lower()], curr_embedding, metric="cosine")
        time_ar = round(time.time() - start_time ,3)
        if distance > 0.85:
            out = {"Auth": False , "Sim": float(distance), "Request" : None, "func": None, "arg" : None, "time ar" :time_ar,  "time trans" : None , "time_gpt" : None , "toggle" : False }
            return jsonify(out)

    # -----------------

    start_time = time.time()
    result = whispher_model.transcribe(temp.name)
    time_trans = round(time.time() - start_time , 3)

    out = {
        "Auth": audio_auth,
        "Sim": float(distance),
        "Request": result['text'],
        "func": None,
        "arg": None,
        "time ar": time_ar,
        "time trans": time_trans,
        "time_gpt": None,
        "toggle" : False
    }

    if "dance" in result["text"].lower():
        out = {
            "Auth": True,
            "Sim": float(distance),
            "Request": result['text'],
            "func": "Dance",
            "arg": None,
            "time ar": time_ar,
            "time trans": time_trans,
            "time_gpt": None,
            "toggle" : False
            }
        return out
    
    if "llm" in result["text"].lower() and "toggle" in result["text"].lower():
        out = {
            "Auth": True,
            "Sim": float(distance),
            "Request": result['text'],
            "func": "llm_res",
            "arg": None,
            "time ar": time_ar,
            "time trans": time_trans,
            "time_gpt": None,
            "toggle" : True
            }
        return out
    
    if "chat" in result["text"].lower() and "gpt" in result["text"].lower() and "toggle" in result["text"].lower():
        out = {
            "Auth": True,
            "Sim": float(distance),
            "Request": result['text'],
            "func": "chatgpt_res",
            "arg": None,
            "time ar": time_ar,
            "time trans": time_trans,
            "time_gpt": None,
            "toggle" : True
            }
        return out
    
    

    # -----------------
    start_time = time.time()
    if response_method == "chatGPT":
        out['func'], out['arg'] = gptResponse(
            "Please Provide answer in two or three sentences " + result['text']
        )
            
    elif response_method == "LLM":
        print("started LLM")
        llm_res = LLMResponse( result['text'] )
        print("LLM done")
        
        out['func'], out['arg'] = "chat" , llm_res 
            
    time_gpt = round(time.time() - start_time , 3)
    out['time_gpt'] =  time_gpt

    return jsonify(out)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5002)
