from flask import Flask, abort, request, jsonify , Response
from tempfile import NamedTemporaryFile
from helper_chatgpt import gptResponse
import time
from ultralytics import YOLO

AUDIO_RECOG = False
FACE_RECOG = True
TRANSCRIBE = True
EMOTION = False
WAKE_WORD = True
FALL_DETECTION = True
IMITATE = True
from LLM_code.LLM_llama3 import LLMResponse

#from LLM_code.LLM1 import LLMResponse  # for gpt
if FALL_DETECTION:
    from ultralytics import YOLO
    from flask import request, Response, Flask
    from PIL import Image
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from pydantic import BaseModel
    from sklearn.preprocessing import LabelEncoder
    import pickle
    import time
    yolo_model_fall = YOLO('models/yolov8n-pose_fall.pt')
    model_inference =  RandomForestClassifier()
    with open('models/fall_detection_classifier.pkl', 'rb') as f:
        model_inference = pickle.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('models/fall_detection_classes.npy', allow_pickle=True)

    class GetKeypoint(BaseModel):
        NOSE:           int = 0
        LEFT_EYE:       int = 1
        RIGHT_EYE:      int = 2
        LEFT_EAR:       int = 3
        RIGHT_EAR:      int = 4
        LEFT_SHOULDER:  int = 5
        RIGHT_SHOULDER: int = 6
        LEFT_ELBOW:     int = 7
        RIGHT_ELBOW:    int = 8
        LEFT_WRIST:     int = 9
        RIGHT_WRIST:    int = 10
        LEFT_HIP:       int = 11
        RIGHT_HIP:      int = 12
        LEFT_KNEE:      int = 13
        RIGHT_KNEE:     int = 14
        LEFT_ANKLE:     int = 15
        RIGHT_ANKLE:    int = 16
    
    def keypoints_to_orderedList(keypoints, conf):
        get_keypoint = GetKeypoint()
        nose_x, nose_y, nose_s = keypoints[get_keypoint.NOSE][0], keypoints[get_keypoint.NOSE][1], conf[get_keypoint.NOSE]
        left_eye_x, left_eye_y, left_eye_s = keypoints[get_keypoint.LEFT_EYE][0], keypoints[get_keypoint.LEFT_EYE][1], conf[get_keypoint.LEFT_EYE]
        right_eye_x, right_eye_y, right_eye_s = keypoints[get_keypoint.RIGHT_EYE][0], keypoints[get_keypoint.RIGHT_EYE][1], conf[get_keypoint.RIGHT_EYE]
        left_ear_x, left_ear_y, left_ear_s = keypoints[get_keypoint.LEFT_EAR][0], keypoints[get_keypoint.LEFT_EAR][1], conf[get_keypoint.LEFT_EAR]
        right_ear_x, right_ear_y, right_ear_s = keypoints[get_keypoint.RIGHT_EAR][0], keypoints[get_keypoint.RIGHT_EAR][1], conf[get_keypoint.RIGHT_EAR]
        left_shoulder_x, left_shoulder_y, left_shoulder_s = keypoints[get_keypoint.LEFT_SHOULDER][0], keypoints[get_keypoint.LEFT_SHOULDER][1], conf[get_keypoint.LEFT_SHOULDER]
        right_shoulder_x, right_shoulder_y, right_shoulder_s = keypoints[get_keypoint.RIGHT_SHOULDER][0], keypoints[get_keypoint.RIGHT_SHOULDER][1], conf[get_keypoint.RIGHT_SHOULDER]
        left_elbow_x, left_elbow_y, left_elbow_s = keypoints[get_keypoint.LEFT_ELBOW][0], keypoints[get_keypoint.LEFT_ELBOW][1], conf[get_keypoint.LEFT_ELBOW]
        right_elbow_x, right_elbow_y, right_elbow_s = keypoints[get_keypoint.RIGHT_ELBOW][0], keypoints[get_keypoint.RIGHT_ELBOW][1], conf[get_keypoint.RIGHT_ELBOW]
        left_wrist_x, left_wrist_y, left_wrist_s = keypoints[get_keypoint.LEFT_WRIST][0], keypoints[get_keypoint.LEFT_WRIST][1], conf[get_keypoint.LEFT_WRIST]
        right_wrist_x, right_wrist_y, right_wrist_s = keypoints[get_keypoint.RIGHT_WRIST][0], keypoints[get_keypoint.RIGHT_WRIST][1], conf[get_keypoint.RIGHT_WRIST]
        left_hip_x, left_hip_y, left_hip_s = keypoints[get_keypoint.LEFT_HIP][0], keypoints[get_keypoint.LEFT_HIP][1], conf[get_keypoint.LEFT_HIP]
        right_hip_x, right_hip_y, right_hip_s = keypoints[get_keypoint.RIGHT_HIP][0], keypoints[get_keypoint.RIGHT_HIP][1], conf[get_keypoint.RIGHT_HIP]
        left_knee_x, left_knee_y, left_knee_s = keypoints[get_keypoint.LEFT_KNEE][0], keypoints[get_keypoint.LEFT_KNEE][1], conf[get_keypoint.LEFT_KNEE]
        right_knee_x, right_knee_y, right_knee_s = keypoints[get_keypoint.RIGHT_KNEE][0], keypoints[get_keypoint.RIGHT_KNEE][1], conf[get_keypoint.RIGHT_KNEE]
        left_ankle_x, left_ankle_y, left_ankle_s = keypoints[get_keypoint.LEFT_ANKLE][0], keypoints[get_keypoint.LEFT_ANKLE][1], conf[get_keypoint.LEFT_ANKLE]
        right_ankle_x, right_ankle_y, right_ankle_s = keypoints[get_keypoint.RIGHT_ANKLE][0], keypoints[get_keypoint.RIGHT_ANKLE][1], conf[get_keypoint.RIGHT_ANKLE]
        return np.array([nose_x,nose_y,nose_s,left_eye_x,left_eye_y,left_eye_s,right_eye_x,right_eye_y,right_eye_s,left_ear_x,left_ear_y,left_ear_s,right_ear_x,right_ear_y,right_ear_s,left_shoulder_x,left_shoulder_y,left_shoulder_s,right_shoulder_x,right_shoulder_y,right_shoulder_s,left_elbow_x,left_elbow_y,left_elbow_s,right_elbow_x,right_elbow_y,right_elbow_s,left_wrist_x,left_wrist_y,left_wrist_s,right_wrist_x,right_wrist_y,right_wrist_s,left_hip_x,left_hip_y,left_hip_s,right_hip_x,right_hip_y,right_hip_s,left_knee_x,left_knee_y,left_knee_s,right_knee_x,right_knee_y,right_knee_s,left_ankle_x,left_ankle_y,left_ankle_s,right_ankle_x,right_ankle_y,right_ankle_s])
    # app = Flask(__name__)

if IMITATE:
    import cv2
    import copy
    import time
    import random
    import numpy as np
    import numpy as np
    from ultralytics import YOLO
    import cv2
    import math
    import base64
    from flask import request, Response, Flask, abort
    import json
    from utils import angle_detection
    from tempfile import NamedTemporaryFile
    from collections import defaultdict

    yolo_model_imitate = YOLO('models/yolov8n-pose_imitate.pt')
    def offset_angles(body_angles):
        if body_angles:
            body_angles[0] = body_angles[0] - 90
            if body_angles[1] <0:
                body_angles.append(0)
                body_angles[1] = -1*body_angles[1]
            else:
                body_angles.append(100)
            if body_angles[3] <0:
                body_angles.append(0)
                body_angles[3] = -1*body_angles[3]
            else:
                body_angles.append(100)
            body_angles[1] = - (180 - body_angles[1])
            body_angles[2] = - (body_angles[2] - 90)
            body_angles[3] = 180 - body_angles[3]
        return body_angles

    def pose_keypoints(image):
        max_width = -1
        human_box = None
        index = -1
        results = yolo_model_imitate(image)
        if results[0]:
            for i, box in enumerate(results[0].boxes.xywh):
                if max_width < box[2]:
                    max_width = box[2]
                    human_box = box
                    index = i
            if human_box is None:
                return None
            return(results[0].keypoints[index].xy[0])
        return None

    def get_angle_in_radians(angle, leg_angles):
        if angle is not None:
            body_angle_in_radians = [math.radians(x) for x in angle[:4]]
            for i, el in enumerate(body_angle_in_radians):
                if math.isnan(el):
                    body_angle_in_radians[i] = 0
            # pitch is the rotation around the shoulder socket
            direction = [] # -119.5 to 119.5
            direction.append(math.radians(0)) if int(angle[-2]) > 50 else direction.append(math.radians(180))
            direction.append(math.radians(0)) if int(angle[-1]) > 50 else direction.append(math.radians(180))
            body_angle_in_radians = direction + body_angle_in_radians
            if body_angle_in_radians[0] < -2.0857: body_angle_in_radians[0] = -2.0857
            if body_angle_in_radians[0] > 2.0857: body_angle_in_radians[0] = 2.0857
            if body_angle_in_radians[1] < -2.0857: body_angle_in_radians[1] = -2.0857
            if body_angle_in_radians[1] > 2.0857: body_angle_in_radians[1] = 2.0857
            if body_angle_in_radians[2] < 0.01: body_angle_in_radians[2] = 0.01 # for making sure hand is straight and not going back
            if body_angle_in_radians[2] > 1.5620: body_angle_in_radians[2] = 1.5620
            if body_angle_in_radians[3] < -1.5620: body_angle_in_radians[3] = -1.5620
            if body_angle_in_radians[3] > -0.0087: body_angle_in_radians[3] = -0.0087
            if body_angle_in_radians[4] < -1.5620: body_angle_in_radians[4] = -1.5620
            if body_angle_in_radians[4] > -0.01: body_angle_in_radians[4] = -0.01
            if body_angle_in_radians[5] < 0.0087: body_angle_in_radians[5] = 0.0087
            if body_angle_in_radians[5] > 1.5620: body_angle_in_radians[5] = 1.5620
            for angle in leg_angles:
                body_angle_in_radians.append(angle)
            return body_angle_in_radians
        return None

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
    #import timm

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
    from facenet_pytorch.models.mtcnn import MTCNN

    face_recog_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_recog_app.prepare(ctx_id=0, det_size=(640, 640))
    #mtcnn_detector = MTCNN(margin=0, thresholds=[0.85,0.95,0.95], device='cuda:0')
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
    (x1,y1) = face[0],face[1]#int(face['bbox'][0]), int(face['bbox'][1])
    (x2,y2) = face[2],face[3]#int(face['bbox'][2]) , int(face['bbox'][3]) 

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

if FALL_DETECTION:
    def detect_objects_on_image(buf):
        """
        Function receives an image,
        passes it through YOLOv8 neural network
        and returns an array of detected objects
        and their bounding boxes
        :param buf: Input image file stream
        :return: Array of bounding boxes in format 
        [[x1,y1,x2,y2,detection],..]
        """
        results = yolo_model_fall(buf)
        output = []
        # try:
        #     for result in results:
        #     # Draw the bounding box on the frame
        #         index = -1
        #         for keypoints in result.keypoints.xyn.cpu().numpy():
        #             index+=1
        #             box = result.boxes[index]
        #             x1, y1, x2, y2 = [
        #             round(x) for x in box.xyxy[0].tolist()
        #             ]
        #             X = keypoints_to_orderedList(keypoints,result.keypoints.conf.cpu().numpy()[index])
        #             try:
        #                 out = model_inference.predict([X])
        #                 detection = encoder.classes_[out][0]
        #             except:
        #                 detection="No Detection"
        #             output.append([
        #             x1, y1, x2, y2 , detection
        #             ])
        # except:
        #     pass
        for result in results:
            # Draw the bounding box on the frame
            index = -1
            for keypoints in result.keypoints.xyn.cpu().numpy():
                index+=1
                box = result.boxes[index]
                x1, y1, x2, y2 = [
                round(x) for x in box.xyxy[0].tolist()
                ]
                X = keypoints_to_orderedList(keypoints,result.keypoints.conf.cpu().numpy()[index])
                try:
                    out = model_inference.predict([X])
                    detection = encoder.classes_[out][0]
                except:
                    detection="No Detection"
                output.append([
                x1, y1, x2, y2 , detection
                ])
        return output
    
    @app.route("/fall_detect", methods=["POST"])
    def detect():
        """
            Handler of /detect POST endpoint
            Receives uploaded file with a name "image_file", 
            passes it through YOLOv8 object detection 
            network and returns an array of bounding boxes.
            :return: a JSON array of objects bounding 
            boxes in format 
            [[x1,y1,x2,y2,object_type,probability],..]
        """
        # start_time = time.time()
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        boxes = detect_objects_on_image(image)
        # print("FPS: ", 1.0 / (time.time() - start_time))
        return Response(
        json.dumps(boxes),  
        mimetype='application/json'
        )
    
if IMITATE:
    @app.route("/imitate", methods=['POST'])
    def imitate():
        if not request.files:
            abort(400)
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # try:
        keypoints = pose_keypoints(image)

        if keypoints is None:
            _, buffer = cv2.imencode('.jpg', image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return {'angles': None, 'image': img_str}
        keypoints = keypoints.cpu()
        body_angles = angle_detection.get_body_angles(keypoints)
        leg_angles = angle_detection.get_leg_angles(keypoints)
        body_angles = offset_angles(body_angles)
        angles = get_angle_in_radians(body_angles, leg_angles)
        # print(angles)

        # annotated_image = image.copy()
        if angles:
            for x, y in keypoints:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return {"angles": angles, "image": img_str}
        # except Exception as e:
        #     _, buffer = cv2.imencode('.jpg', image)
        #     img_str = base64.b64encode(buffer).decode('utf-8')
        #     return {'angles': None, 'image': img_str}
    
if FACE_RECOG:
    @app.route('/face_recog', methods=['POST'])
    def face_recog():
        if not request.files:
            abort(400)

        # Handle the image data
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        faces = face_recog_app.get(image)
        # boxes,_ = mtcnn_detector.detect(image)
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
                    image = app.draw_on(image, [faces[i]] )
        # if boxes is not None:
        #     for box in boxes:
        #         x,y,w,h = box
        #         x1,y1 = int(x), int(y)
        #         x2,y2= int(x1+w), int(y1+h)
        #         label = None
                if EMOTION:
                    label = give_emotion( [x1,y1,x2,y2] , image)

                print(faces[i]['bbox'])
                (x1,y1) = int(faces[i]['bbox'][0]), int(faces[i]['bbox'][1])
                (x2,y2) = int(faces[i]['bbox'][2]) , int(faces[i]['bbox'][3])
                image = face_recog_app.draw_on_mod(image, [faces[i]], detected_p , label , str(round(accu , 2)) , color)
                
                cv2.rectangle(image, (x1,y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)

                # Display the label and accuracy percentage
                text = f"{label}"
                cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        
        # cv2.imwrite('processed_image.jpg', image)
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
    
    if ("nao" in result["text"].lower() or "now" in result["text"].lower()) and "toggle" in result["text"].lower():
        out = {
            "Auth": True,
            "Sim": float(distance),
            "Request": result['text'],
            "func": "AIKO",
            "arg": None,
            "time ar": time_ar,
            "time trans": time_trans,
            "time_gpt": None,
            "toggle" : True
            }
        return out
    
    if ("kai" in result["text"].lower() or "pepper" in result["text"].lower() or "paper" in result["text"].lower()) and "toggle" in result["text"].lower():
        out = {
            "Auth": True,
            "Sim": float(distance),
            "Request": result['text'],
            "func": "KAI",
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
