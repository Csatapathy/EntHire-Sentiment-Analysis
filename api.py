# Some Necessary Imports
from wsgiref.simple_server import make_server

from pyramid.config import Configurator
from pyramid.response import Response

from inference import Inference


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
    inference.data = data  # sets data in inference class's constructor
    result = inference.get_predictions()  # makes predictions and inferences
    print("Sent JSONified results....")
    return result


if __name__ == '__main__':

    # loads the inference class that makes predictions and inferences
    inference = Inference()

    with Configurator() as config:
        # include OpenAPI for Swagger Documentation
        config.include("pyramid_openapi3")
        config.pyramid_openapi3_spec("api_docs/swagger.yaml")
        config.pyramid_openapi3_add_explorer()

        config.add_route('start', '/home')  # Route for the starting page
        config.add_view(starting_page, 'start')

        config.add_route('p', '/predict')  # Route for the predictions API
        config.add_view(get_predictions, route_name="p",
                        renderer="json", openapi=True)

        config.scan(".")
        app = config.make_wsgi_app()  # Make the API

    print("API Started....Check out:",
          "\nhttp://localhost:6543/home for the starting page",
          "\nhttp://localhost:6543/predict for the API ",
          "\nhttp://localhost:6543/docs/ for the Swagger Documentation")

    # http://localhost:6543/ as the address
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
