import requests


def get_valuation(code="sh600519"):
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        resp = requests.get(url)
        resp.encoding = "gbk"
        data = resp.text.split('"')[1].split('~')

        if len(data) < 47:
            raise ValueError("返回数据字段不足")

        return {
            "市盈率": float(data[39]) if data[39] else None,
            "市净率": float(data[46]) if data[46] else None,
            "总市值": float(data[45]) if data[45] else None,
        }
    except Exception as e:
        print(f"[警告] 获取腾讯估值数据失败: {e}")
        return {
            "市盈率": None,
            "市净率": None,
            "总市值": None,
        }