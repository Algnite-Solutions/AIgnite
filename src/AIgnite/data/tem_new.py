import base64
from volcengine.visual.VisualService import VisualService
import json

if __name__ == '__main__':
    visual_service = VisualService()
    # call below method if you dont set ak and sk in $HOME/.volc/config
    visual_service.set_ak('')
    visual_service.set_sk('')

    params = dict()

    form = {
        "image_base64":  base64.b64encode(open("/data3/peirongcan/paperIgnite/AIgnite/src/AIgnite/data/test.pdf",'rb').read()).decode(),   # 文件binary 图片/PDF 
        "image_url": "",                  # url
        "version": "v3",                  # 版本
        "page_start": 0,                  # 起始页数
        "page_num": 16,                   # 解析页数
        "table_mode": "html",             # 表格解析模式
        "filter_header": "true"           # 过滤页眉页脚水印
    }

    # 请求
    resp = visual_service.ocr_pdf(form)

    if resp["data"]:
        markdown = resp["data"]["markdown"] # markdown 字符串
        json_data = resp["data"]["detail"] # json格式详细信息

        with open("/data3/peirongcan/paperIgnite/AIgnite/src/AIgnite/data/resp.md", "w") as f:
            f.writelines(markdown)

        json_data = json.loads(json_data)

        # 保存json
        with open("/data3/peirongcan/paperIgnite/AIgnite/src/AIgnite/data/resp.json", "w") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    else:
        print("request error")