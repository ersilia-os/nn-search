FROM golang:1.25.1-bookworm AS build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libfreetype6-dev libpng-dev zlib1g-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o /out/server .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libfreetype6 libpng16-16 zlib1g libstdc++6 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/server .
EXPOSE 8080
ENV MILVUS_ADDR=milvus-standalone:19530
USER 10001
ENTRYPOINT ["/app/server"]

