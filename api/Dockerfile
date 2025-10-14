FROM golang:1.25 AS build
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /app/server .

FROM gcr.io/distroless/base-debian12
WORKDIR /app
COPY --from=build /app/server .
EXPOSE 8080
ENV MILVUS_ADDR=milvus-standalone:19530
USER nonroot:nonroot
ENTRYPOINT ["/app/server"]
