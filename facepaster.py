# https://github.com/tejonaco/facepaster

from facenet_pytorch import MTCNN
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
from math import asin, cos, degrees, radians


# OPTIONS
# overscaling face
X_SCALE_MARGIN = 1.1
Y_SCALE_MARGIN = 1.1


# start the face detector
mtcnn = MTCNN()

# initialize a cache dict to improve perfomance
# avoindig use mtcnn method, wich is slow in less powerfull devices
# cache is also persistent using pickle
from hashlib import md5
import pickle
try:
    with open('cache', 'rb') as f:
        cache = pickle.load(f) # md5: [untransformed_box, ut_landmarks, box, size]
except FileNotFoundError:
    cache = {}


class BadFaceError(Exception):
    pass


class FacePaster:
    """
    Recomended usage as library:
    with FacePaster(input_face) as fp:
        img = fp.paste_faces(input_img)
        fp.plot(img) #optional
    """
    
    def __init__(self, input_face: str or BytesIO):
        # load input face
        try:
            self.input_face = Image.open(input_face)
        except UnidentifiedImageError:
            raise BadFaceError('Document is not a image')

        # check if it is in buffer, if it is exit of the function before slow _get_features()
        hash_ = md5(self.input_face.tobytes()).hexdigest()
        if hash_ in cache:
            ut_box, ut_landmarks, self.if_box, self.if_size = cache[hash_]
        else:
            [ut_box], _, [ut_landmarks] = self._get_features(self.input_face) # if not in cache proceed to the nn

        # remodelate face using untransformed box and landmarks
        # make face y-axis aligned
        degrees = self._get_rotation(ut_landmarks) # get current inclination
        self.input_face = self.input_face.rotate(-degrees) # rotate in the oposite way

        # make face left looking
        direction = self._get_direction(ut_box, ut_landmarks)
        if direction == 'right':
            self.input_face = self.input_face.transpose(Image.FLIP_LEFT_RIGHT)

        # if face isn't in cache recalculate params and save in cache
        if hash_ not in cache:
            # recalculate params once face has been remodelated
            try:
                [self.if_box], [self.if_size], [landmarks] = self._get_features(self.input_face)
            except ValueError:
                raise BadFaceError('You must send one face as input image')

            # save face in cache
            cache[hash_] = ut_box, ut_landmarks, self.if_box, self.if_size
            with open('cache', 'wb') as f: #update the persistent cache
                pickle.dump(cache, f)
                

    def __enter__(self):
        return self


    def __exit__(self, *ex_args):
        self.input_face.close()

    
    @staticmethod
    def _get_features(img: Image) -> tuple:
        """Returns data from mtcnn output"""
        img_formated = np.array(img.convert('RGB'))
        boxes, probs, landmarks = mtcnn.detect(img_formated, landmarks=True)
        sizes = [(box[0] - box[2], box[1] - box[3]) for box in boxes]
        return boxes, sizes, landmarks

    
    def _reshape_input_face(self, face_size: tuple) -> Image:
        face_h, face_w, _ = np.array(self.input_face).shape
        escale_x, escale_y = face_size[0] / self.if_size[0] * X_SCALE_MARGIN, face_size[1] / self.if_size[1] * Y_SCALE_MARGIN
        new_size = int(face_w * escale_x), int(face_h * escale_y)
        return self.input_face.resize(new_size, Image.ANTIALIAS)

    @staticmethod
    def _get_rotation(landmarks: list) -> float:
        """returns rotation degrees based on eye's landmarks"""
        eye_l, eye_r = landmarks[:2]
        x, y = (eye_r[0] - eye_l[0]), (eye_l[1] - eye_r[1])
        return degrees(asin(y / x))

    
    def _get_paste_position(self, box: list, size: tuple, degrees: float=0) -> tuple:
        x, y = box[:2]
        x_correction = self.if_box[0] * (size[0] / self.if_size[0]) / cos(radians(degrees)) * X_SCALE_MARGIN
        y_correction = self.if_box[1] * (size[1] / self.if_size[1]) / cos(radians(degrees)) * Y_SCALE_MARGIN

        x -= int(x_correction)
        y -= int(y_correction)

        return int(x), int(y)

    @staticmethod  
    def _get_direction(box: list, landmarks: list) -> str:
        """Computes distance between the box and both mouth edges to determine direction"""
        mouth_l, mouth_r = landmarks[:2, 0]
        box_l, box_r = box[0], box[2]
        if (mouth_l - box_l) > (box_r - mouth_r):
            return 'right'
        else:
            return 'left'

    
    def paste_faces(self, img_path: str or BytesIO) -> Image:
        try:
            with Image.open(img_path) as img:
                for box, size, landmarks in zip(*self._get_features(img)):
                    # get x direction
                    direction = self._get_direction(box, landmarks)
                    # get face and flip it if it's necessary
                    face = self._reshape_input_face(size)
                    if direction == 'right':
                        face = face.transpose(Image.FLIP_LEFT_RIGHT)
                    # get rotation and rotate face
                    degrees = self._get_rotation(landmarks)
                    face = face.rotate(degrees)
                    # gets best position to face and paste it
                    position = self._get_paste_position(box, size, degrees)
                    img.paste(face, position, mask=face)

                return img
        except TypeError:
            return img # if image does not contain any faces return the same

    
    @classmethod
    def plot(cls, img: Image, plot_rectangles: bool=False, plot_landmarks: bool = False) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        # Create figure and axes
        fig,ax = plt.subplots(1)
        # compute features
        boxes, sizes, landmarks = cls._get_features(img)
        # plot background image
        plt.imshow(np.array(img))

        for box, landmark in zip(boxes, landmarks):
            rect_shape = box[:2], box[2] - box[0], box[3] - box[1]
            if plot_rectangles:
                rect = Rectangle(*rect_shape, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            if plot_landmarks:
                ax.scatter(landmark[:, 0], landmark[:, 1], s=8)

        plt.show()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Paste a input face on the input image faces')

    parser.add_argument('face', metavar='face', type=str, nargs=1, help='input face')
    parser.add_argument('img', metavar='img', type=str, nargs=1, help='input image')
    parser.add_argument('-o', dest='output', nargs=1, help='(optional) output file')
    parser.add_argument('-p', action='store_const', const=True, help='(optional) plot image')

    args = parser.parse_args()

    with FacePaster(*args.face) as fp:
        img = fp.paste_faces(*args.img)
        if args.p or not args.output:
            fp.plot(img)
        if args.output:
            img.save(*args.output)
