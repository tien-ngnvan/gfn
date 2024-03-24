# Build stage:
FROM hieupth/mamba:pypy3 AS build

ADD . .
RUN apt-get update && \
    mamba install -c conda-forge conda-pack && \
    mamba env create -f environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "gfn", "/bin/bash", "-c"]
# 
RUN pip install .
#
RUN conda-pack -n gfn -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
#
RUN /venv/bin/conda-unpack

# Runtime stage:
FROM debian:buster AS runtime
# Copy /venv from the previous stage:
COPY --from=build /venv /venv
#
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && uvicorn gfn.api:app --host 0.0.0.0 --port 8080