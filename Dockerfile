FROM python:3.10-slim

WORKDIR app/

RUN pip install --upgrade pip

COPY . .

RUN pip install .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
