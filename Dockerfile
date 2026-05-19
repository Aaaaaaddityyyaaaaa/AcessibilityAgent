FROM python:3.13-slim

WORKDIR /backend

COPY src/ ./src/
COPY model/feature_model/f_model.pkl.dvc ./model/feature_model/
COPY artifacts/model/decoder.pkl.dvc ./artifacts/model/
COPY data/image_preprocess.py ./data/
COPY data/__init__.py ./data/
COPY api/ ./api/
COPY config.yaml .
COPY pyproject.toml .
COPY requirements.txt .
COPY .dvc/config ./.dvc/config

RUN pip install -r requirements.txt
RUN pip install -e .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
dvc remote modify origin --local auth basic\n\
dvc remote modify origin --local user $DAGSHUB_USERNAME\n\
dvc remote modify origin --local password $DAGSHUB_TOKEN\n\
dvc pull\n\
exec uvicorn api.api:app --host 0.0.0.0 --port 7860' > /entrypoint.sh \
&& chmod +x /entrypoint.sh

EXPOSE 7860

CMD ["/entrypoint.sh"]