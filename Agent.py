# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.

import numpy as np
from RavensFigure import RavensFigure
from RavensProblem import RavensProblem
import cv2
import time
from PIL import Image
from PIL import ImageChops

class Agent:
    def __init__(self):
        pass
    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.

    def Solve(self, problem):
        start = time.time()
        self.answer_image = []
        self.image_1 = cv2.imread(problem.figures['1'].visualFilename)
        ret, thresh1 = cv2.threshold(self.image_1, 127, 255, cv2.THRESH_BINARY)
        self.answer_image.append(self.image_1)

        self.image_2 = cv2.imread(problem.figures['2'].visualFilename)
        ret, thresh1 = cv2.threshold(self.image_2, 127, 255, cv2.THRESH_BINARY)
        self.answer_image.append(self.image_2)

        self.image_3 = cv2.imread(problem.figures['3'].visualFilename)
        ret, thresh1 = cv2.threshold(self.image_3, 127, 255, cv2.THRESH_BINARY)
        self.answer_image.append(self.image_3)

        self.image_4 = cv2.imread(problem.figures['4'].visualFilename)
        ret, thresh1 = cv2.threshold(self.image_4, 127, 255, cv2.THRESH_BINARY)
        self.answer_image.append(self.image_4)

        self.image_5 = cv2.imread(problem.figures['5'].visualFilename)
        ret, thresh1 = cv2.threshold(self.image_5, 127, 255, cv2.THRESH_BINARY)
        self.answer_image.append(self.image_5)

        self.image_6 = cv2.imread(problem.figures['6'].visualFilename)
        ret, thresh1 = cv2.threshold(self.image_6, 127, 255, cv2.THRESH_BINARY)
        self.answer_image.append(self.image_6)


        if problem.problemSetName == "Basic Problems B" or problem.problemSetName == "Test Problems B" or problem.problemSetName == "Challenge Problems B" or problem.problemSetName == "Raven's Problems B":

            self.image_A = cv2.imread(problem.figures['A'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_A, 127, 255, cv2.THRESH_BINARY)
            self.image_B = cv2.imread(problem.figures['B'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_B, 127, 255, cv2.THRESH_BINARY)
            self.image_C = cv2.imread(problem.figures['C'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_C, 127, 255, cv2.THRESH_BINARY)
            index_ = 0
            if self.calculate(self.image_A, self.image_B) > 0.999:
                for value in self.answer_image:
                    index_ += 1
                    if self.calculate(self.image_C, value) > 0.999:
                        return index_
            if self.calculate(self.image_A, self.image_C) > 0.999:
                for value in self.answer_image:
                    index_ += 1
                    if self.calculate(self.image_B, value) > 0.999:
                        return index_
            DPR_list = []
            IPR_list = []
            DPR_AB = self._DPR(self.image_A, self.image_B)
            IPR_AB = self._IPR(self.image_A, self.image_B)
            for ele in self.answer_image:
                DPR = self._DPR(self.image_C, ele)
                DPR_list.append(DPR)
            for ele in self.answer_image:
                IPR = self._IPR(self.image_C, ele)
                IPR_list.append(IPR)

            DPR_diff_list =[]
            IPR_diff_list =[]
            for ele in DPR_list:
                DPR_diff = abs(ele - DPR_AB)
                DPR_diff_list.append(DPR_diff)
            for ele in IPR_list:
                IPR_diff = abs(ele - IPR_AB)
                IPR_diff_list.append(IPR_diff)
            candidate_result_1 = min(DPR_diff_list)
            candidate_result_2 = min(IPR_diff_list)
            result = min(candidate_result_1, candidate_result_2)
            if result == candidate_result_1:
                index = DPR_diff_list.index(candidate_result_1)
                return index + 1
            else:
                index = IPR_diff_list.index(candidate_result_2)
                return index + 1

        else:
            DPR_Answer_AG = []
            DPR_Answer_GH = []
            DPR_Answer_DG = []
            self.image_F = cv2.imread(problem.figures['F'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_F, 127, 255, cv2.THRESH_BINARY)
            self.image_H = cv2.imread(problem.figures['H'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_H, 127, 255, cv2.THRESH_BINARY)
            self.image_G = cv2.imread(problem.figures['G'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_G, 127, 255, cv2.THRESH_BINARY)
            self.image_C = cv2.imread(problem.figures['C'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_C, 127, 255, cv2.THRESH_BINARY)
            self.image_A = cv2.imread(problem.figures['A'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_A, 127, 255, cv2.THRESH_BINARY)
            self.image_D = cv2.imread(problem.figures['D'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_D, 127, 255, cv2.THRESH_BINARY)
            self.image_B = cv2.imread(problem.figures['B'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_B, 127, 255, cv2.THRESH_BINARY)
            self.image_7 = cv2.imread(problem.figures['7'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_7, 127, 255, cv2.THRESH_BINARY)
            self.answer_image.append(self.image_7)

            self.image_8 = cv2.imread(problem.figures['8'].visualFilename)
            ret, thresh1 = cv2.threshold(self.image_8, 127, 255, cv2.THRESH_BINARY)
            self.answer_image.append(self.image_8)


            if problem.problemSetName == "Basic Problems D" or problem.problemSetName == "Test Problems D" or problem.problemSetName == "Challenge Problems D" or problem.problemSetName == "Raven's Problems D":
                DPR_GH = self._DPR(self.image_G, self.image_H)
                DPR_AG = self._DPR(self.image_A, self.image_G)
                DPR_DG = self._DPR(self.image_D, self.image_G)
                for element in self.answer_image:
                    DPR_Answer_AG.append(abs(self._DPR(self.image_C, element) - DPR_AG))
                for element in self.answer_image:
                    DPR_Answer_DG.append(abs(self._DPR(self.image_F, element) - DPR_DG))
                for element in self.answer_image:
                    DPR_Answer_GH.append(abs(self._DPR(self.image_H, element) - DPR_GH))
                value1 = min(DPR_Answer_AG)
                value2 = min(DPR_Answer_GH)
                value3 = min(DPR_Answer_DG)
                value = min(value1, value2, value3)
                if value == value1:
                    index = DPR_Answer_AG.index(value)
                elif value == value2:
                    index = DPR_Answer_GH.index(value)
                else:
                    index = DPR_Answer_DG.index(value)
                end = time.time()
                duration = end - start
                print(duration)
                return index + 1


            if problem.problemSetName == "Basic Problems C" or problem.problemSetName == "Test Problems C" or problem.problemSetName == "Challenge Problems C" or problem.problemSetName == "Raven's Problems C":
                DPR_GH = self._DPR(self.image_G, self.image_H)
                DPR_AG = self._DPR(self.image_A, self.image_G)
                DPR_DG = self._DPR(self.image_D, self.image_G)
                IPR_GH = self._IPR(self.image_G, self.image_H)
                for element in self.answer_image:
                    DPR_Answer_AG.append(abs(self._DPR(self.image_C, element) - DPR_AG))
                for element in self.answer_image:
                    DPR_Answer_DG.append(abs(self._DPR(self.image_F, element) - DPR_DG))

                #calculate all the DPR and IPR
                DPR_Hi_list = []
                IPR_Hi_list = []
                candidate_IPR = []
                IPR_com_list = []
                for image in self.answer_image:
                    DPR_Hi_list.append(self._DPR(self.image_H, image))
                    IPR_Hi_list.append(self._IPR(self.image_H, image))

                low_limit = DPR_GH * 0.4
                upper_limit = DPR_GH * 1.6

                for ele in DPR_Hi_list:
                    if low_limit <= ele <= upper_limit:
                        candidate_IPR.append(IPR_Hi_list[DPR_Hi_list.index(ele)])
                if len(candidate_IPR) != 0:
                    for i in candidate_IPR:
                        IPR_com_list.append(abs(i - IPR_GH))

                    answer_IPR_com = min(IPR_com_list)
                    temp_index = IPR_com_list.index(answer_IPR_com)
                    answer_IPR = candidate_IPR[temp_index]
                    index = IPR_Hi_list.index(answer_IPR)
                else:
                    for element in self.answer_image:
                        DPR_Answer_GH.append(abs(self._DPR(self.image_H, element) - DPR_GH))
                    value1 = min(DPR_Answer_AG)
                    value2 = min(DPR_Answer_GH)
                    value3 = min(DPR_Answer_DG)
                    value = min(value1, value2, value3)
                    if value == value1:
                        index = DPR_Answer_AG.index(value)
                    elif value == value2:
                        index = DPR_Answer_GH.index(value)
                    else:
                        index = DPR_Answer_DG.index(value)





                end = time.time()
                duration = end - start
                print(duration)
                return index + 1


            if problem.problemSetName == "Basic Problems E" or problem.problemSetName == "Test Problems E" or problem.problemSetName == "Challenge Problems E" or problem.problemSetName == "Raven's Problems E":
                A_or_B = cv2.bitwise_or(self.image_A, self.image_B)
                A_and_B = cv2.bitwise_and(self.image_A, self.image_B)
                temp = cv2.bitwise_xor(self.image_A, self.image_B)
                A_xor_B = cv2.bitwise_not(temp)

                _or_ = self.compare(A_or_B, self.image_C)
                _and_ = self.compare(A_and_B, self.image_C)
                _xor_ = self.compare(A_xor_B, self.image_C)


                selected_pattern = max(_or_, _and_, _xor_)
                answer_score = []
                if selected_pattern == _or_:
                    G_or_H = cv2.bitwise_or(self.image_G, self.image_H)
                    for answer in self.answer_image:
                        answer_score.append(self.compare(G_or_H, answer))

                elif selected_pattern == _and_:
                    G_and_H = cv2.bitwise_and(self.image_G, self.image_H)
                    for answer in self.answer_image:
                        answer_score.append(self.compare(G_and_H, answer))

                elif selected_pattern == _xor_:
                    G_xor_H_temp = cv2.bitwise_xor(self.image_G, self.image_H)
                    G_xor_H = cv2.bitwise_not(G_xor_H_temp)
                    for answer in self.answer_image:
                        answer_score.append(self.compare(G_xor_H, answer))


                socre = max(answer_score)
                answer = answer_score.index(socre) + 1
                end = time.time()
                duration = end - start
                print(duration)
                return answer



    def _DPR(self, image1, image2):
        dpr_1 = np.sum(image1 == 0)
        dpr_2 = np.sum(image2 == 0)
        dpr = abs(dpr_1 / (dpr_2+1))
        return dpr

    def compare(self, img1, img2):
        result = np.count_nonzero((img1 == 0) & (img2 == 0)) / (
                    np.count_nonzero((img1 == 0) & (img2 == 0)) + np.count_nonzero(img1 == 0) + np.count_nonzero(img2 == 0))

        return result


    def _IPR(self, image1, image2):
        result_image = cv2.bitwise_or(image1, image2)
        result_pixels = np.sum(result_image)
        IPR_1 = result_pixels/np.sum(image1)
        IPR_2 = result_pixels/np.sum(image2)
        return IPR_1 - IPR_2

# reference: https://www.cnblogs.com/lld76/p/15995217.html
    def calculate(self, image1, image2):
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1
        degree = degree / len(hist1)
        return degree