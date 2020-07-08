# Flickr로 사진 검색해서 다운로드하기
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint

import os, time, sys

# AP 키 지정하기 ---( % 1)
key = "68cf7a5d518edd6fd3e1e2f38521951f"
secret = "b0574c0cc2fd4752"
wait_time=.5 # 대기 시간 (초)

# 키워드와 디렉터리 이름 지정해서 다운로드하기 -- (% 2)
def main():
    go_download('오징어', 'squid')
    go_download('치킨', 'chicken')
    go_download('케이크', 'cake')

# Flickr API로 사진검색하기 -- ( %3)
def go_download(keyword, dir):

    # 저장 경로 지정하기
    savedir = "./flickr/" + dir 
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # API를 사용해서 다운로드하기 ---
    flickr = FlickrAPI(key, secret, format='parsed-json')
    res = flickr.photos.search(
        text = keyword,          # 키워드
        per_page = 350,          # 검색할 개수
        media = 'photos',        # 사진 검색
        sort = "relevance",      # 키워드 관련도 순서
        safe_search = 1,         # 안전 검색
        extras = 'url_n, license'
    )

    # 결과 확인하기
    photos = res['photos']
    pprint(photos)
    try:
        # 1장씩 다운로드하기 -- ( %5)
        for i, photo in enumerate(photos['photo']):
            url_n = photo['url_n']
            filepath = savedir + '/' + photo['id'] + '.jpg'
            if os.path.exists(filepath): continue
            print(str(i+1) + ":download=", url_n)
            urlretrieve(url_n, filepath)
            time.sleep(wait_time)
    except:
        import traceback
        traceback.print_exc()


if __name__== '__main__':  # https://dojang.io/mod/page/view.php?id=2448
    main()

    
'''

https://medium.com/@adrianmrit/creating-simple-image-datasets-with-flickr-api-2f19c164d82f

SIZES = ["url_o", "url_k", "url_h", "url_l", "url_c"]
url_o: Original (4520 × 3229)
url_k: Large 2048 (2048 × 1463)
url_h: Large 1600 (1600 × 1143)
url_l=: Large 1024 (1024 × 732)
url_c: Medium 800 (800 × 572)
url_z: Medium 640 (640 × 457)
url_m: Medium 500 (500 × 357)
url_n: Small 320 (320 × 229)
url_s: Small 240 (240 × 171)
url_t: Thumbnail (100 × 71)
url_q: Square 150 (150 × 150)
url_sq: Square 75 (75 × 75)
'''