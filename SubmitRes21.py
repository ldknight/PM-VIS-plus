'''
    用于动态提交results.zip 文件
    @该文件仅支持ytvis2021
    python SubmitRes21.py --descriptionname IDOL_YTVIS21_swinL_masklossANDBoxinstloss_trackRes_IDOLBoxinstSwinLInferenceMask_gtBox_ytvis21_vi_DelRepeatDelOutboxpart_maskloss05model_0074999 --filepathname /share/home/liudun/VNext-main/outdir/IDOL_YTVIS21_swinL_masklossANDBoxinstloss_trackRes_IDOLBoxinstSwinLInferenceMask_gtBox_ytvis21_vi_DelRepeatDelOutboxpart_maskloss05/inference/model_0074999.pth/results.zip
'''

import urllib.request
import urllib.parse
import json
import requests
from urllib import parse
from xml.dom import minidom
import os, random
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse
parser = argparse.ArgumentParser() ## 新建参数解释器对象
parser.add_argument('--descriptionname',
        default="test submit",
        help="文件描述") ## 添加参数,注明参数类型
parser.add_argument('--filepathname',
                    default="",
                    metavar="FILE",
                    help="文件路径"
                    ) ## 添加参数
args = parser.parse_args()### 参数赋值，也可以通过终端赋值

'''
    第一个获取网页详情
'''
def get_papers():
    #第一个网址的GET请求
    url = 'https://codalab.lisn.upsaclay.fr/competitions/6064/submissions/9134?_=1676859377336'
    headers = {
        'accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.5',
        'Referer': 'https://codalab.lisn.upsaclay.fr/competitions/6064',
        'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Microsoft Edge";v="108"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': 'macOS',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'Cookie': 'messages="00964e81dc061d51fd34732123929895f61ddc00$[[\"__json_message\"\0540\05425\054\"Successfully signed in as goodeinston.\"]]"; csrftoken=Ihl7XhEDQd1ssPqsMrNRAN7h7UE6QX8oUbYDq9OarSVxRbKZ8kJUFb3mlzbqjdcq; sessionid=r66jn04t32lu6oiraxduhckd6hgdtcyc',
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54",
        "x-requested-with": "XMLHttpRequest"
    }
    # 在请求头中带上User-Agent，模拟浏览器发送请求
    response = requests.get(url, headers=headers)
    return response.content
'''
    第二个上传参数请求post
'''
def get_upload_params():
    url = 'https://codalab.lisn.upsaclay.fr/s3direct/get_upload_params/'
    headers = {
        'accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'origin': 'https://codalab.lisn.upsaclay.fr',
        'referer': 'https://codalab.lisn.upsaclay.fr/competitions/7680',
        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': 'macOS',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'cookie': 'messages="00964e81dc061d51fd34732123929895f61ddc00$[[\"__json_message\"\0540\05425\054\"Successfully signed in as goodeinston.\"]]"; csrftoken=Ihl7XhEDQd1ssPqsMrNRAN7h7UE6QX8oUbYDq9OarSVxRbKZ8kJUFb3mlzbqjdcq; sessionid=r66jn04t32lu6oiraxduhckd6hgdtcyc',
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.41",
        "x-csrftoken": "Ihl7XhEDQd1ssPqsMrNRAN7h7UE6QX8oUbYDq9OarSVxRbKZ8kJUFb3mlzbqjdcq"
    }
    # post请求参数
    param = {
        'type': 'application/zip',
        'name': 'results.zip',
        'dest': 'submissions'
    }
    # post请求参数需要进行两次编码，第一次urlencode：对字典参数进行Unicode编码转成字符串，第二次encode：将字符串数据转换为字节类型
    param = urllib.parse.urlencode(param).encode('utf-8')
    # post定制请求可以使用位置传参
    request = urllib.request.Request(url, param, headers)
    response = urllib.request.urlopen(request)
    # 解码读取数据
    page = response.read().decode('utf-8')
    # 反序列化，将字节对象转成python对象
    content = json.loads(page)
    # print(content)
    return content
'''第3个 上传文件的请求 post'''
def py3_private(file_path_name="/Users/liudun/Desktop/results.zip"):
    get_upload_params_res = get_upload_params()
    post_url = get_upload_params_res['form_action']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.41',
        'Referer': post_url
    }
    target_fields = get_upload_params_res
    target_fields['success_action_status'] = '201'
    target_fields['file'] = (os.path.basename(file_path_name), open(file_path_name, 'rb'), 'application/octet-stream')

    multipart_encoder = MultipartEncoder(
        fields=target_fields,
        boundary='-----------------------------' + str(random.randint(1e28, 1e29 - 1))
    )
    headers['Content-Type'] = multipart_encoder.content_type
    # 请求头必须包含一个特殊的头信息，类似于Content-Type: multipart/form-data; boundary=${bound}
    r = requests.post(post_url, data=multipart_encoder, headers=headers)
    return r.text
'''第4个 submission'''
def submission(id,description_name="123333333444555566"):
    url = 'https://codalab.lisn.upsaclay.fr/api/competition/7680/submission'
    headers = {
        'accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'cookie': 'messages="00964e81dc061d51fd34732123929895f61ddc00$[[\"__json_message\"\0540\05425\054\"Successfully signed in as goodeinston.\"]]"; csrftoken=Ihl7XhEDQd1ssPqsMrNRAN7h7UE6QX8oUbYDq9OarSVxRbKZ8kJUFb3mlzbqjdcq; sessionid=r66jn04t32lu6oiraxduhckd6hgdtcyc',
        'origin': 'https://codalab.lisn.upsaclay.fr',
        'referer': 'https://codalab.lisn.upsaclay.fr/competitions/7680',
        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': 'macOS',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.41",
        "x-csrftoken": "Ihl7XhEDQd1ssPqsMrNRAN7h7UE6QX8oUbYDq9OarSVxRbKZ8kJUFb3mlzbqjdcq",
        'x-requested-with': 'XMLHttpRequest'
    }
    params = {
        'description': description_name,
        'docker_image': None,
        'method_name': None,
        'method_description': None,
        'project_url': None,
        'publication_url': None,
        'team_name': None,
        'organization_or_affiliation': None,
        'bibtex': None,
        'phase_id': 12638
    }
    data = {
        'id': id,
        'name': None,
        'type': None,
        'size': None
    }
    data = parse.urlencode(data)
    response04 = requests.post(url=url,
                                  headers=headers,
                                  params=params,
                                  data=data)
    return response04.content.decode('utf-8')
if __name__ == "__main__":
    description_name = args.descriptionname
    file_path_name = args.filepathname
    # print(description_name)
    # print(file_path_name)

    # description_name = "testtest"
    # file_path_name="/Users/liudun/Desktop/personSearchGuidance/results/idol_ytvis2019_r50_boxinst_lossproj/inference/model_0004999.pth/results.zip"
    py3_private_res =py3_private(file_path_name=file_path_name)
    # print(py3_private_res)
    doc = minidom.parseString(py3_private_res)
    name_arr = doc.getElementsByTagName("Location")
    if len(name_arr):
        # print(name_arr[0].firstChild.data)
        submission_res = submission(id=str(name_arr[0].firstChild.data), description_name=description_name)
        print(submission_res)
        print("submitted!!!")
    else:
        print("py3_private_res error!!!")

class SubmitRes(object):
    """multipart/form-data格式转化"""
    @staticmethod
    def submit(description_name="xxx",file_path_name=""):
        # description_name = "testtest"
        # file_path_name = "/Users/liudun/Desktop/personSearchGuidance/results/idol_ytvis2019_r50_boxinst_lossproj/inference/model_0004999.pth/results.zip"
        py3_private_res = py3_private(file_path_name=file_path_name)
        print(py3_private_res)
        doc = minidom.parseString(py3_private_res)
        name_arr = doc.getElementsByTagName("Location")
        if len(name_arr):
            print(name_arr[0].firstChild.data)
            submission_res = submission(id=str(name_arr[0].firstChild.data), description_name=description_name)
            print(submission_res)
            print("submitted!!!")
        else:
            print("py3_private_res error!!!")
