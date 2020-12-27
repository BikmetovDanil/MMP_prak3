Третье задание практикума, декабрь 2020.

Добавлены ансамбли моделей, серверная реализация и Docker. В папке Server/test лежат тестовые файлы: train.csv для обучения, target.csv - ответы к обучающему файлу, test.csv - файл для тестирования predict, test_answer.csv - файл с ответами на test.

На сервере происходит автоматическое удаление признаков, которые невозможно сравнить.

Команды для Docker:

	docker build -t ml_server .
	
	docker run --rm -p 5000:5000 -v "$PWD/Server/data:/root/Server/data" --rm -i ml_server

