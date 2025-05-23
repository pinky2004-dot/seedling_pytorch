from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        file_path = f"data/{file.filename}"
        file.save(file_path)

        response = requests.post("http://127.0.0.1:7000/predict/", files={"file": open(file_path, "rb")})
        prediction = response.json()["prediction"]

        return render_template("result.html", prediction=prediction)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5000)