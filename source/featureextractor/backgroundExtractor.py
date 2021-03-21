import utils
import cv2
import argparse
import backgroundFeatureInfo as background

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()

def backgroundExtractor(imageName):
    image = cv2.imread(imageName)
    background.determineBackground(image, args.image, args.file)

# backgroundExtractor(args.image)

#utils.getBackgroundFeatureInfo('asdf', 'asdf')
