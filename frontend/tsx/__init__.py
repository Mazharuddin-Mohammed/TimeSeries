import tsx_backend

class ARIMA:
    def __init__(self, p=1, d=0, q=0):
        self.model = tsx_backend.ARIMA(p, d, q)

    def fit(self, data):
        return self.model.fit(data)

if __name__ == "__main__":
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    model = ARIMA(p=2, d=1, q=0)
    params = model.fit(data)
    print(f"Estimated ARIMA parameters: {params}")