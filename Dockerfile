FROM python:3.8
RUN pip install pandas scikit-learn streamlit pickle
COPY src/app.py /app/
COPY model/svm_model.pkl /app/model/svm_model.pkl
COPY data/coches.csv /app/
WORKDIR /app

ENTRYPOINT ["streamlit", "run", "app.py"]

