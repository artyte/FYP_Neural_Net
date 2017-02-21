import csv

class WikiHistory:

	testData = []
	developData = []
	trainData = []

	@staticmethod
	def __init__():
		with open("wiki_history_write.csv", encoding = "utf8") as f0:
			read = csv.reader(f0)
			row_count = sum(1 for row in read)
			f0.seek(0)	#reset readline to start for other reading purposes
			WikiHistory.prepareTrainingData(read, row_count)
			WikiHistory.prepareDevelopmentData(read, row_count)
			WikiHistory.prepareTestData(read, row_count)

	@staticmethod
	def prepareTrainingData(input, row_count):
		for index, row in enumerate(input):
			if (index < int(row_count * 0.8)):
				WikiHistory.trainData.append([row[2], row[3]])
			else:
				WikiHistory.developData.append([row[2], row[3]])
				break

	@staticmethod
	def prepareDevelopmentData(input, row_count):
		for index, row in enumerate(input):
			if (index < int(row_count * 0.1) - 1):
				WikiHistory.developData.append([row[2], row[3]])
			else:
				WikiHistory.testData.append([row[2], row[3]])
				break

	@staticmethod
	def prepareTestData(input, row_count):
		for index, row in enumerate(input):
			WikiHistory.testData.append([row[2], row[3]])

	@staticmethod
	def getLabels(string):
		lis = []
		if (string == "TESTING_DATA"):
			for row in WikiHistory.testData:
				lis.append(row[1])
			return lis
		elif (string == "DEVELOPMENT_DATA"):
			for row in WikiHistory.developData:
				lis.append(row[1])
			return lis
		elif (string == "TRAINING_DATA"):
			for row in WikiHistory.trainData:
				lis.append(row[1])
			return lis

	@staticmethod
	def getFeatures(string):
		lis = []
		if (string == "TESTING_DATA"):
			for row in WikiHistory.testData:
				lis.append(row[0])
			return lis
		elif (string == "DEVELOPMENT_DATA"):
			for row in WikiHistory.developData:
				lis.append(row[0])
			return lis
		elif (string == "TRAINING_DATA"):
			for row in WikiHistory.trainData:
				lis.append(row[0])
			return lis

	@staticmethod
	def getList(string):
		lis = []
		if (string == "TESTING_DATA"):
			return WikiHistory.testData
		elif (string == "DEVELOPMENT_DATA"):
			return WikiHistory.developData
		elif (string == "TRAINING_DATA"):
			return WikiHistory.trainData
