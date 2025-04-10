import tsx
import pytest

def test_arima_fit():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = tsx.ARIMA(p=1, d=0, q=0)
    params = model.fit(data)
    assert len(params) == 1