from flask import Flask, request
# Run live trader as app is started

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    """ Return a friendly HTTP greeting. """
    who = request.args.get("who", "World")
    import live_trader
    live_trader
    return f"Hello {who}!\n"

# app.run(host="localhost", port=8080, debug=True)

if __name__ == "__main__":
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host="localhost", port=8080, debug=False)
    import live_trader
    live_trader
