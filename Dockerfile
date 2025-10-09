# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=1 \
    TZ=UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

COPY pyproject.toml README.md ./
COPY a22a ./a22a
COPY configs ./configs
COPY staged ./staged
COPY tests ./tests

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip wheel --no-deps -w /tmp/wheels .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=1 \
    TZ=UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

COPY --from=builder /tmp/wheels /tmp/wheels

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

COPY pyproject.toml README.md ./
COPY a22a ./a22a
COPY configs ./configs
COPY staged ./staged
COPY tests ./tests

VOLUME ["/app"]

EXPOSE 8501

CMD ["streamlit", "run", "a22a/reports/app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
