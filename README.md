№Запуск

docker-compose up --build


Все финальные файлы будут выполнены в корректном порядке


# Docker Compose

docker-compose up --build

docker-compose up -d --build

docker-compose logs -f

# Docker (Direct)

docker build -t fraud-detection:latest .

docker run -v %cd%/data:/app/data -v %cd%/outputs:/app/outputs fraud-detection:latest
