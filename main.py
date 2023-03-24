# coding: utf-8

import argparse
import image
import naivefusion
import laplacianfusion

# Loading the arguments

# parser를 이용해 입력 받은 문자열을 문자 단위의 토큰으로 분해하고 
# 분해된 문자를 기준으로 값 전달
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# -l, --list
# text파일의 경로(?)를 갖는 str 값을 받으며
# 해당 text 파일은 이미지들의 경로를 포함한다.
# 기본값은 'list_images.txt'
parser.add_argument(
    '-l',
    '--list',
    dest='names',
    type=str,
    default='list_images.txt',
    help='The text file which contains the names of the images')

# -f, --folder
# 반드시 포함해야하는 값
# 이미지를 포함하는 폴더 경로
# 위 -l 값이 파일 이름만 갖는다면 이건 폴더 경로까지만
# 기본으로 imageset 폴더까지 들어감
parser.add_argument(
    '-f',
    '--folder',
    dest='folder',
    type=str,
    required=True,
    help='The folder containing the images')

# -hp, --heightpyr
# 정수값
# 라플라시안 피라미드의 층 갯수 = 축적 정도 2의 5제곱까지 축적
# 1, 1/2, 1/4, 1/8, 1/16, 1/32
parser.add_argument(
    '-hp',
    '--heightpyr',
    dest='height_pyr',
    type=int,
    default=6,
    help='The height of the Laplacian pyramid')

# -wc
# 실수값
# contrast 지수
parser.add_argument(
    '-wc',
    dest='w_c',
    type=float,
    default=1.0,
    help='Exponent of the contrast')

# -ws
# 실수값
# saturation 지수
parser.add_argument(
    '-ws',
    dest='w_s',
    type=float,
    default=1.0,
    help='Exponent of the saturation')

# -we
# 실수값
# exposedness 지수
parser.add_argument(
    '-we',
    dest='w_e',
    type=float,
    default=1.0,
    help='Exponent of the exposedness')

# contrast 차이
# saturation 포화
# exposedness 노출
# 3개의 가중치를 선별하는 애들이었음
args = parser.parse_args()
params = vars(args)  # convert to ordinary dict

names = [line.rstrip('\n') for line in open(params['names'])]
folder = params['folder']
height_pyr = params['height_pyr']
w_c = params['w_c']
w_s = params['w_s']
w_e = params['w_e']

# Naive Fusion

W = naivefusion.WeightsMap(folder, names)
res_naive = W.result_exposure(w_c, w_s, w_e)
image.show(res_naive)

# Laplacian Fusion

lap = laplacianfusion.LaplacianMap(folder, names, n=height_pyr)
res_lap = lap.result_exposure(w_c, w_s, w_e)
image.show(res_lap)
