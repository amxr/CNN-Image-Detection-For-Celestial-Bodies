import os
from base64 import b64encode
from flask import render_template, request, redirect, url_for, send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, validators
import keras
import numpy as np
from app.__init__ import app
from hub.examples.image_retraining.label_image import wiki
from hub.examples.image_retraining.reverse_image_search import reverseImageSearch
import keras.utils as image
from werkzeug.datastructures import FileStorage
import wikipedia
from yaml import load, SafeLoader

# no secret key set yet
SECRET_KEY = os.urandom(32)
app.config["SECRET_KEY"] = SECRET_KEY
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_RETRAINING = os.path.abspath(os.path.join(app.root_path, "..", "hub", "examples", "image_retraining"))
MODEL = os.path.join(IMAGE_RETRAINING, "CNN_Model.h5")
UPLOADS = os.path.join(APP_ROOT, 'uploads')
app.config['UPLOADED_PHOTOS_DEST'] = UPLOADS
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class SelectImageForm(FlaskForm):
    image_url = StringField(
        "image_url",
        validators=[validators.Optional(), validators.URL()],
        render_kw={"placeholder": "Enter a URL"},
    )
    image_file = FileField(
        "file",
        validators=[
            validators.Optional(),
            FileAllowed(["jpg", "jpeg", "png"], "Invalid File"),
        ],
        render_kw={"class": "custom-file-input"},
    )


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(UPLOADS, filename)


@app.route("/upload")
def upload():
    form = SelectImageForm()
    if form.validate_on_submit():
        image_file = FileStorage(
            stream=form.image_file.data.stream,
            filename=form.image_file.data.filename,
            content_type=form.image_file.data.content_type,
            content_length=form.image_file.data.content_length,
            headers=form.image_file.data.headers,
        )

        # save the image to the UPLOADS_DEFAULT_DEST folder
        filename = photos.save(image_file)
        file_url = url_for('get_file', filename=filename)
        predict_answer()


@app.route("/", methods=["GET", "POST"])
def index():
    # Get a list of all the files in the folder
    files = os.listdir(UPLOADS)
    if len(files) != 0:
        # Loop through the list of files and remove each one
        for file_name in files:
            file_path = os.path.join(UPLOADS, file_name)
            os.remove(file_path)
    # image in memory will be used on reload
    global imageBytes
    form = SelectImageForm()
    if form.validate_on_submit():
        if request.files.get(form.image_file.name):
            upload()
        return redirect(url_for("result"))
    print("returning to index")
    return render_template("index.html", form=form)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/result")
def result():
    try:
        # Get a list of all the files in the directory
        files = os.listdir(UPLOADS)

        # Get the first file in the directory
        first_file = files[0]
        # Open the file in binary mode and read its contents
        with open(os.path.join(UPLOADS, first_file), 'rb') as f:
            image_data = f.read()

        # Convert the binary data to a base64 string
        base64_data = b64encode(image_data).decode('utf-8')
    except Exception as e:
        return render_template("error.html", detail=str(e))

    celestial_object, labels = predict_answer()
    title, properties, description = wiki(celestial_object, IMAGE_RETRAINING)
    return render_template(
        "result.html",
        image=base64_data,
        labels=labels,
        title=title,
        description=description,
        properties=properties,
    )


@app.route("/redirect_to_google")
def redirect_to_google():
    search_url = reverseImageSearch(imageBytes)
    return redirect(search_url, 302)


@app.route("/predict")
def predict_answer():
    files = os.listdir(UPLOADS)
    # Get the first file in the directory
    first_file = files[0]
    if first_file:
        # Get a list of all the files in the directory
        files = os.listdir(UPLOADS)

        # Get the first file in the directory
        first_file = files[0]
        first_file_path = os.path.join(UPLOADS, first_file)

        img_width = 256
        img_height = 256

        img = image.load_img(first_file_path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # normalize pixel values to [0, 1]
        model = keras.models.load_model(MODEL)
        # Make the prediction
        predictions = model.predict(x)

        # Print the predicted class label and probability
        class_names = ['asteroids', 'earth', 'elliptical', 'jupiter', 'mars', 'mercury', 'moon', 'neptune', 'saturn',
                       'spiral', 'uranus', 'venus']
        predicted_class_index = np.argmax(predictions)
        print(predictions)
        print(predicted_class_index)
        predicted_class_label = class_names[predicted_class_index]
        predicted_probability = predictions[0][predicted_class_index]
        print(f"The predicted class is {predicted_class_label} with probability {predicted_probability:.2f}")

        predictions_100 = [[p * 100 for p in row] for row in predictions]
        predictions = dict(zip(class_names, predictions_100[0]))

        labels_and_scores = list(predictions.items())
        # Print the dictionary
        print(predictions)
        print("results page")
        return predicted_class_label, labels_and_scores


# return title, statistics and summary
def wiki(celestial_object, cwd):
    ans = celestial_object
    with open(os.path.join(cwd, "display_info.yml"), "r") as stream:
        all_display_statistics = load(stream, Loader=SafeLoader)

    req_statistics = all_display_statistics.get(ans, {})
    statistics = None
    title = None
    summary = None
    if ans in ["spiral", "elliptical"]:
        title = ("Classified Celestial Object is {} Galaxy : ".format(ans.capitalize()))
        summary = wikipedia.WikipediaPage(title="{} galaxy".format(ans)).summary
    elif ans in [
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ]:
        title = ("Classified Celestial Object is {} Planet : ".format(ans.capitalize()))
        statistics = req_statistics.items()
        summary = wikipedia.WikipediaPage(title="{} (planet)".format(ans)).summary
    elif ans == "moon":
        statistics = req_statistics.items()
        summary = wikipedia.WikipediaPage(title="{}".format(ans)).summary
        title = ("Classified Celestial Object is the {} : ".format(ans.capitalize()))
    elif ans == "asteroids":
        statistics = req_statistics.items()
        summary = wikipedia.WikipediaPage(title="{}".format(ans)).summary
        title = ("Classified Celestial Object is the {} : ".format(ans.capitalize()))
    return title, statistics, summary
