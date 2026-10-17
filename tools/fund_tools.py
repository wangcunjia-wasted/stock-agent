from data.valuation import get_valuation


def calc_fundamental(code="sh600519"):
    val = get_valuation(code)
    pe = val["市盈率"]
    pb = val["市净率"]

    if pe is None:
        valuation = "未知"
    elif pe < 15:
        valuation = "偏低"
    elif pe < 30:
        valuation = "合理"
    else:
        valuation = "偏高"

    return {
        "市盈率": pe,
        "市净率": pb,
        "总市值": val["总市值"],
        "估值定性": valuation
    }