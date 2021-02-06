from operator import itemgetter
import cv2
import os, shutil


class Sort:
    def __init__(self):
        self.rows = []
        self.final_cord = []
        self.temp_cord = []
        self.pre = 0

    def __del__(self):
        print("Destroyed")

    def rowAlign(self, coordinates_text):
        with open(coordinates_text) as f:
            lines = f.readlines()

        for i in lines:
            res = [int(i) for i in i[:-1].split(',')]

            if self.pre != 0 and (res[1] - self.pre) > 5:
                self.rows.append(self.temp_cord)
                self.temp_cord = []
                self.temp_cord.append(res)

            else:
                self.temp_cord.append(res)

            self.pre = res[1]

    def finalAlign(self):
        for i in self.rows:
            columns_sort = sorted(i, key=itemgetter(0))
            self.final_cord.append(columns_sort)

    def imageCrop(self, original_image, name_of_folder):
        os.mkdir('./Text_Detection/crop_images/' + name_of_folder)
        img = cv2.imread(original_image)
        a = 1
        for i in self.final_cord:
            b = 1
            for j in i:
                print(str(a) + "." + str(b))
                print(j)
                crop_img = img[j[1]:j[5], j[0]:j[2]]
                try:
                    cv2.imwrite('./Text_Detection/crop_images/' + name_of_folder + "/" + str(a) + "." + str(b) + '.jpg',
                                crop_img)
                except Exception as e:
                    print(e)
                b += 1

            # print()
            a += 1



