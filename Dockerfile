FROM python:3.13

RUN mkdir /yournewface
ADD requirements.txt /yournewface/

RUN pip3 install -r /yournewface/requirements.txt

EXPOSE 8080

COPY . /yournewface/
WORKDIR /yournewface