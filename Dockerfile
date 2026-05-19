FROM python:3.13-slim

WORKDIR /backend

COPY src/ ./src/

COPY model/feature_model/f_model.pkl.dvc ./model/feature_model/

COPY artifacts/model/decoder.pkl.dvc ./artifacts/model/

COPY data/image_preprocess.py ./data/
COPY data/__init__.py ./data/

COPY api/ ./api/

COPY config.yaml . 

COPY pyproject.toml  .

COPY requirements.txt .

COPY .dvc/config ./.dvc/config

ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN

RUN pip install -r requirements.txt

RUN pip install -e .

RUN dvc remote modify origin --local auth basic && \
    dvc remote modify origin --local user ${DAGSHUB_USERNAME} && \
    dvc remote modify origin --local password ${DAGSHUB_TOKEN}

RUN dvc pull

EXPOSE 8000

CMD [ "uvicorn" , "api.api:app" , "--host" ,"0.0.0.0" , "--port" ,"8000"]

