# To test locally run: docker build -t test_image .

FROM ubuntu:18.04
WORKDIR /var/app/tropical
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3-pip
RUN apt-get install -y libpq-dev
RUN apt-get install -y net-tools wget cmake
RUN apt-get install -y libsm6

COPY docker_requirements.txt /tmp

RUN python3 -m pip install -r /tmp/docker_requirements.txt

COPY . /var/app/

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader vader_lexicon
RUN python3 -m nltk.downloader averaged_perceptron_tagger
RUN python3 -m nltk.downloader maxent_ne_chunker
RUN python3 -m nltk.downloader words
RUN python3 -m nltk.downloader tagsets
RUN python3 -m nltk.downloader stopwords

RUN python3 -m spacy download en

EXPOSE 7100
EXPOSE 9100
CMD ["python3", "rest_flask_app.py", "--server_port=7100", "--prometheus_port=9100"]