Третье задание практикума, декабрь 2020.

Добавлены ансамбли моделей, серверная реализация и Docker. В папке Server/test лежат тестовые файлы: train.csv для обучения, target.csv - ответы к обучающему файлу, test.csv - файл для тестирования predict, test_answer.csv - файл с ответами на test.

На сервере происходит автоматическое удаление признаков, которые невозможно привести к float.

Команды для Docker:

	docker pull bikmetovdanil/mmp_prak3
	
	docker run --rm -p 5000:5000 -i bikmetovdanil/mmp_prak3

