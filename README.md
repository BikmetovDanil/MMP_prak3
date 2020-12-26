Третье задание практикума, декабрь 2020
Добавлены ансамбли, серверная реализация и Docker

Команды для Docker:

	docker build -t flask_server .
	
	docker run --rm -p 5000:5000 -v "$PWD/FlaskExample/data:/root/Server/data" --rm -i flask_server

