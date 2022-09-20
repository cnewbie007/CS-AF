import numpy as np
from sklearn.preprocessing import MinMaxScaler


class GetCostM:

    def __init__(self, severe_score):
        self.severe_score = severe_score
        self.size = len(self.severe_score)
        self.cost_matrix = np.zeros((self.size, self.size))

    def compute_relative_cost(self):
        for row in range(self.size):
            for col in range(self.size):
                # Diagonal values indicate the severe score
                if col == row:
                    self.cost_matrix[row, col] = 9 - self.severe_score[row]
                # Compute the relative cost based on the severe score
                else:
                    self.cost_matrix[row, col] = self.severe_score[row] ** 2 / self.severe_score[col] ** 2

    def rescale(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.cost_matrix[row, col] < 1:
                    self.cost_matrix[row, col] = self.cost_matrix[col, row] * 0.6

    def minmax_norm(self):

        temp = np.zeros((56, 1))
        k = 0
        for row in range(self.size):
            for col in range(self.size):
                if row != col:
                    temp[k] = self.cost_matrix[row, col]
                    k += 1
        scaler = MinMaxScaler(feature_range=(2 * max(self.severe_score), 200))
        scaler.fit(temp)
        cost = scaler.transform(temp)

        k = 0
        for row in range(self.size):
            for col in range(self.size):
                if row != col:
                    self.cost_matrix[row, col] = np.round(cost[k], 0)
                    k += 1

    def view_save(self):
        print(self.cost_matrix)
        print(self.cost_matrix.T)
        # np.savetxt('./cost_matrix/cost_matrix_B.txt', self.cost_matrix, delimiter=',')
        # np.savetxt('./cost_matrix/cost_matrix_A.txt', self.cost_matrix.T, delimiter=',')

    def workflow(self):

        self.compute_relative_cost()

        self.rescale()

        self.minmax_norm()

        self.view_save()


if __name__ == '__main__':

    severe = [1, 4, 3, 5, 8, 6, 7, 2]
    GetCostM(severe).workflow()
