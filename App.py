from Text_Detection.test import text_detect
from Text_Detection.sort_image_crop import Sort
from Text_Recognization import demo
import os, shutil
from flask import Flask, request, Response, jsonify

app = Flask(__name__)


@app.route('/uploadImages', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        files = request.files.getlist("images")

        if not os.path.isdir('./Text_Detection/data'):
            os.mkdir('./Text_Detection/data')
        else:
            shutil.rmtree('./Text_Detection/data')
            os.mkdir('./Text_Detection/data')

        if not os.path.isdir('./Text_Detection/result'):
            os.mkdir('./Text_Detection/result')
        else:
            shutil.rmtree('./Text_Detection/result')
            os.mkdir('./Text_Detection/result')

        if not os.path.isdir('./Text_Recognization/demo_images'):
            os.mkdir('./Text_Recognization/demo_images')
        else:
            shutil.rmtree('./Text_Recognization/demo_images')
            os.mkdir('./Text_Recognization/demo_images')

        try:
            for f in files:
                full_path = './Text_Detection/data/' + str((f.filename))
                f.save(full_path)

            return jsonify({'message': 'images uploaded!'})

        except FileExistsError as fee:
            return jsonify({'message': str(fee)})


@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    text_detect()

    if not os.path.isdir('./Text_Detection/crop_images'):
        os.mkdir('./Text_Detection/crop_images')
    else:
        shutil.rmtree('./Text_Detection/crop_images')
        os.mkdir('./Text_Detection/crop_images')

    for i in os.listdir('Text_Detection/data'):
        sort = Sort()
        sort.rowAlign('./Text_Detection/result/res_' + str(i).split(".")[0] + '.txt')
        sort.finalAlign()
        sort.imageCrop('./Text_Detection/data/' + str(i), str(i).split(".")[0])
        del sort

    # Text recognize line by line and get final string
    final_data = []

    for i in os.listdir('./Text_Detection/crop_images'):
        print("Image Name ", i)
        data = {"Images Name": str(i), "whole_text": ""}
        image_names = []
        for j in os.listdir('./Text_Detection/crop_images/' + str(i)):
            image_names.append(j)

        b = [float(sent.split(".jpg")[0]) for sent in image_names]
        b.sort()

        temp_images = []
        for k in range(0, len(b) - 1):
            if (b[k] // 1 - b[k + 1] // 1) == 0.0:
                temp_images.append(b[k])
            else:
                temp_images.append(b[k])
                new_set_of_images = [str(sent) + '.jpg' for sent in temp_images]

                for u in new_set_of_images:
                    shutil.copy('./Text_Detection/crop_images/' + str(i) + "/" + u, './Text_Recognization/demo_images/')

                text_line = demo.textRecognize()
                data["whole_text"] = data["whole_text"] + '\n' + text_line
                shutil.rmtree('./Text_Recognization/demo_images/')
                os.mkdir('./Text_Recognization/demo_images/')
                temp_images = []

        final_data.append(data)
        print(data)
    final_data_json = {'Data': final_data}

    return jsonify({'message': final_data_json})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
