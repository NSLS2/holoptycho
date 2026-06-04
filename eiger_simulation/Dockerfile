FROM python:3.12

ENV POETRY_VERSION=1.8.3
ENV AS_HDF5_MASTER_FILE=/home/asuser/ansto_simplon_api/master_file_examples/scan_257331_raw.h5
ENV AS_NUMBER_OF_DATA_FILES=1

# Install OS packages
USER root
RUN apt-get update
RUN apt-get install -y gcc libhdf5-serial-dev

# Create app runner user
RUN useradd -ms /bin/bash asuser
WORKDIR /home/asuser/

# Setup Poetry environment
RUN git clone https://github.com/AustralianSynchrotron/ansto-simplon-api.git /home/asuser/simplon_repo
RUN cp /home/asuser/simplon_repo/pyproject.toml /home/asuser/
RUN cp /home/asuser/simplon_repo/poetry.lock /home/asuser/
RUN cp /home/asuser/simplon_repo/README.md /home/asuser/

RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install

# Copy source code
RUN cp -r /home/asuser/simplon_repo/ansto_simplon_api /home/asuser/ansto_simplon_api

# copy test data
COPY /test_data/scan_257331_raw.h5 /home/asuser/ansto_simplon_api/master_file_examples/
COPY /*.py /home/asuser/

RUN python /home/asuser/add_dummy_fields.py

USER asuser
EXPOSE 8000 5555
ENTRYPOINT uvicorn ansto_simplon_api.main:app --host 0.0.0.0 --port 8000
