"""Hand-calculated parser cases that guard return units and section boundaries."""


from datetime import date

import pytest

from momenta.reference_data import parse_fred, parse_french, parse_jkp


def test_french_factors_percent_sentinels_and_month_end() -> None:
    text = """This file was created using the 200002 CRSP database.
,Mkt-RF,SMB,HML,RF
200001,10,-99.99,-999,0.5
200002,-10,0,2.5,0.25

Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2000,90,90,90,90
"""
    frame, vintage = parse_french(text, "ff3_monthly")
    assert vintage == "This file was created using the 200002 CRSP database."
    assert frame.to_dicts() == [
        {"date": date(2000, 1, 31), "mkt_rf": 0.1, "smb": None, "hml": None, "rf": 0.005},
        {"date": date(2000, 2, 29), "mkt_rf": -0.1, "smb": 0.0, "hml": 0.025, "rf": 0.0025},
    ]


def test_french_does_not_mix_equal_and_value_weighted_sections() -> None:
    text = """This file was created using the 200001 CRSP database.
Value Weight Returns -- Daily
,Lo PRIOR,PRIOR 2,PRIOR 3,PRIOR 4,PRIOR 5,PRIOR 6,PRIOR 7,PRIOR 8,PRIOR 9,Hi PRIOR
20000103,-10,0,0,0,0,0,0,0,0,20

Equal Weight Returns -- Daily
,Lo PRIOR,PRIOR 2,PRIOR 3,PRIOR 4,PRIOR 5,PRIOR 6,PRIOR 7,PRIOR 8,PRIOR 9,Hi PRIOR
20000103,-90,0,0,0,0,0,0,0,0,90
"""
    frame, _ = parse_french(text, "momentum_deciles_daily")
    assert frame.height == 1
    assert frame["date"].to_list() == [date(2000, 1, 3)]
    assert frame["d1"][0] == -0.1
    assert frame["d10"][0] == 0.2


def test_french_breakpoints_leave_count_unscaled() -> None:
    values = ",".join(str(number) for number in range(5, 101, 5))
    text = f"This file was created using the 200002 CRSP database.\n200002,500,{values}\n"
    frame, _ = parse_french(text, "momentum_breakpoints")
    assert frame["n_stocks"][0] == 500
    assert frame["p05"][0] == 0.05
    assert frame["p100"][0] == 1.0


def test_french_rejects_duplicate_dates_and_missing_vintage() -> None:
    text = "This file was created using the 200002 CRSP database.\n200002,10\n200002,20\n"
    with pytest.raises(ValueError, match="Duplicate dates"):
        parse_french(text, "momentum_monthly")
    with pytest.raises(ValueError, match="Missing CRSP vintage"):
        parse_french("200002,10\n", "momentum_monthly")


def test_jkp_preserves_decimal_return_and_checks_factor_definition() -> None:
    text = """location,name,freq,weighting,direction,n_stocks,n_stocks_min,date,ret
usa,ret_12_1,monthly,vw_cap,1,100,40,2000-02-29,0.125
"""
    frame = parse_jkp(text, "ret_12_1", "monthly")
    assert frame["ret"][0] == 0.125
    assert frame["date"][0] == date(2000, 2, 29)
    with pytest.raises(ValueError, match="Unexpected factor definition"):
        parse_jkp(text.replace("vw_cap", "ew"), "ret_12_1", "monthly")
    with pytest.raises(ValueError, match="Missing factor definition"):
        parse_jkp(text.replace("vw_cap,1", "vw_cap,"), "ret_12_1", "monthly")


def test_fred_preserves_levels_and_missing_days_and_rejects_truncation() -> None:
    payload = {
        "count": 2,
        "observations": [
            {"date": "2000-01-03", "value": "25.5", "realtime_start": "2000-02-01",
             "realtime_end": "2000-02-01"},
            {"date": "2000-01-04", "value": ".", "realtime_start": "2000-02-01",
             "realtime_end": "2000-02-01"},
        ],
    }
    frame = parse_fred(payload)
    assert frame["value"].to_list() == [25.5, None]
    assert frame["realtime_start"].to_list() == [date(2000, 2, 1)] * 2
    payload["count"] = 3
    with pytest.raises(ValueError, match="empty or incomplete"):
        parse_fred(payload)
