FROM python:3.9.5

RUN apt update
RUN python -m pip install --upgrade pip
RUN pip install setuptools==49.6.0
RUN pip install numpy==1.21.0
RUN pip install pandas==1.3.0
RUN pip install scipy==1.7.0
RUN pip install Cython
RUN pip install matplotlib==3.4.2
RUN pip install joblib==1.0.1
RUN pip install threadpoolctl==2.1.0
VOLUME ./project
WORKDIR ./project
COPY . .
WORKDIR ./scikit-learn
RUN pip install cython pytest pytest-cov flake8 mypy black==21.6b0
RUN pip install --no-build-isolation --editable .
WORKDIR ..

CMD python3 scriptTest.py > output_latest.txt