import numpy as np
from itertools import count
import csv
import multiprocessing as mp

def hammingdistance(l1,l2):
	l=np.asarray(l1)-np.asarray(l2)
	l=np.abs(l)
	dis = np.sum(l)
	return dis

class NaiveApproach(object):
	def __init__(self, trait_breed_matrix, reliability_of_worker):
		self.trait_breed_matrix = trait_breed_matrix
		self.no_classes = len(self.trait_breed_matrix)
		self.no_traits = len(trait_breed_matrix[0])
		self.reliability_of_worker=reliability_of_worker
		self.questions_asked = self.no_traits #Since you're asking a trait for every question.
		#self.e=next(e for e in count(1, 1) if (1-self.reliability_of_worker)*gamma(set_sigma([list(range(1,self.no_classes+1))]+[[]]*e))[-1] <= e)
		#self.game_state = [list(range(1,self.no_classes+1))] + [[]]*self.e #Initialising the gamestate as a list of lists.
		self.correct_breed = np.random.randint(1, self.no_classes + 1)
		self.correct_breed_traits = self.trait_breed_matrix[self.correct_breed-1]
		self.answered_breed=self.answer()
	def answer(self):
		response=[]
		for i in range(self.no_traits):
			if self.correct_breed_traits[i] == 1:
				if np.random.uniform(0,1) < self.reliability_of_worker:
					response.append(1)
				else:
					response.append(0)
			elif self.correct_breed_traits[i] == 0:
				if np.random.uniform(0,1) < self.reliability_of_worker:
					response.append(0)
				else:
					response.append(1)
		hamming_distances = list(map(lambda x: hammingdistance(x,response), self.trait_breed_matrix))
		minimum_value = min(hamming_distances)
		likely_answers = [i for i, x in enumerate(hamming_distances) if x == minimum_value]
		return np.random.choice(likely_answers)+1

def generate_breed_trait_matrix(trait_no,breed_no):
	trait_breed=[]
	for _ in range(breed_no):
		successful = False
		while (successful == False):
			rand_breed = np.random.randint(2, size=trait_no).tolist()
			if (rand_breed not in trait_breed):
				successful = True
				trait_breed.append(rand_breed)
	return trait_breed

def naive_sample(i,j,reliability):
	np.random.seed()
	naive = NaiveApproach(generate_breed_trait_matrix(i, j), reliability)
	if naive.answered_breed == naive.correct_breed:
		return (1,naive.questions_asked)
	else:
		return (0,naive.questions_asked)

def naive_parallel_sample(i,j,reliability,e,iter):
	pool=mp.Pool(mp.cpu_count()-2)
	future_res = [pool.apply_async(naive_sample, args=(i,j,reliability)) for _ in range(iter)]
	pool.close()
	res = [f.get() for f in future_res]

	return res

#for some reason this wont work on widows
if __name__ == '__main__':
	reliabilities_to_test=[0.9,0.8]
	times_to_test=1000
	number_of_datapoints=100
	for reliability in reliabilities_to_test:
		#for i in range(9,10):#generate until 16 traits
		i=9
		f = open(('naive_reliability_{0}_data_traits_{1}_{2}_times.csv'.format(reliability, i, times_to_test)), 'a')
		datawriter = csv.writer(f, dialect='excel')
		datawriter.writerow(["trait number", 'breed number', 'expected numbers of errors', 'average number of questions asked','successful answer', 'unsuccessful answer', 'success rate'])
		f.close()
		j=2
		step=8
		#if(2**i/number_of_datapoints<=1):
			#step=1
		#else:
			#step=floor(2**i/number_of_datapoints)
		while(j<=2**i):
			f = open(('naive_reliability_{0}_data_traits_{1}_{2}_times.csv'.format(reliability, i, times_to_test)), 'a')
			datawriter = csv.writer(f, dialect='excel')
			e="Not applicable."
			expected_no=i
			results=naive_parallel_sample(i,j,reliability,e,iter=times_to_test)

			succ_list, ques_no_list = zip(*results)

			successful=np.sum(succ_list)
			unsuccessful=times_to_test-successful
			success_rate=(successful)/times_to_test

			ques_no = np.average(ques_no_list)
			print([i,j,expected_no,e,ques_no,successful,unsuccessful,success_rate])
			datawriter.writerow([i,j,expected_no,e,ques_no,successful,unsuccessful,success_rate])
			print(j)
			j+=step
			f.close()

