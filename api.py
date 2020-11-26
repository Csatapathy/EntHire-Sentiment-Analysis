# Some Necessary Imports
from pathlib import Path
from wsgiref.simple_server import make_server

import joblib
from pyramid.config import Configurator
from pyramid.response import Response

from build_Model import build_model
from model import NLPModel
from inference import Inference


def load_pipeline():
    """
    Loads the pickled pipeline.
    If the pickle isn't there, it builds the model and then proceeds.
    """
    file = Path('NLPModel_joblib.pkl')
    path = 'NLPModel_joblib.pkl'
    if not file.is_file():
        print("There is no pickled model....")
        build_model()
    with open(path, 'rb') as file:
        pipeline = joblib.load(path)
        print("Pipeline loaded from {}".format(path))
    return pipeline


def starting_page(request):
    """
    The starting page response for url : localhost:6543/home
    """
    return Response("Welcome To My Sentiment analysis project! Check out http://localhost:6543/predict for the API.")


def get_predictions(request):
    """
    Takes JSON dict as input and outputs a JSONified string
    """
    print("Making Predictions....")
    data = request.json_body  # Gets the data
    result = {}  # stores the predictions
    inference=Inference()
    for key in data.keys():
        a = data[key]
        predictions = model.predict(a)  # make predictions on the data
        result["predictions"] = inference.make_inference(predictions)  # makes the inferences from the data
    print("Sent JSONified results....")
    return result


if __name__ == '__main__':
    model = NLPModel()  # Load model
    model.pipeline = load_pipeline()  # Load Pipeline from the pickled model
    
    with Configurator() as config:
        config.include("pyramid_openapi3")  # include OpenAPI for Swagger Documentation
        config.pyramid_openapi3_spec("api_docs/swagger.yaml")
        config.pyramid_openapi3_add_explorer()

        config.add_route('start', '/home')  # Route for the starting page
        config.add_view(starting_page, 'start')

        config.add_route('p', '/predict')  # Route for the predictions API
        config.add_view(get_predictions, route_name="p", renderer="json", openapi=True)

        config.scan(".")
        app = config.make_wsgi_app()  # Make the API

    print("API Started....Check out:",
        "\nhttp://localhost:6543/home for the starting page",
        "\nhttp://localhost:6543/predict for the API ",
        "\nhttp://localhost:6543/docs/ for the Swagger Documentation")

    server = make_server('0.0.0.0', 6543, app)  # http://localhost:6543/ as the address
    server.serve_forever()
