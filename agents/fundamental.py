from data.valuation import get_valuation


def analyze_fundamental(code="sh600519"):
    val = get_valuation(code)
    pe = val["市盈率"]
    pb = val["市净率"]

    if pe is None:
        valuation = "估值数据不可用"
    elif pe < 15:
        valuation = "估值偏低"
    elif pe < 30:
        valuation = "估值合理"
    else:
        valuation = "估值偏高"

    return {
        "市盈率": pe,
        "市净率": pb,
        "总市值": val["总市值"],
        "估值判断": valuation,
    }