import pandas as pd
import cv2 as cv

pd.read_csv('out/comparison.csv').to_csv('out/comparison.trim.csv', float_format='%.3f', index=False)
pd.read_csv('out-3/comparison.csv').to_csv('out-3/comparison.trim.csv', float_format='%.3f', index=False)
pd.read_csv('out/comparison.csv').to_latex('out/comparison.tex', index=False, float_format="%.3f")
pd.read_csv('out-3/comparison.csv').to_latex('out-3/comparison.tex', index=False, float_format="%.3f")

img = cv.imread('out/model.png')
rot = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imwrite('out/model-rotated.png', rot)
