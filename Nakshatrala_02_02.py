

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tkinter as tk
import scipy.misc
import os
import mpmath

class DisplayActivationFunctions:


    def __init__(self, root, master, *args, **kwargs):

        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 110
        self.xasis = np.arange(1000)
        self.learning_rate = 0.001
        self.weight_matrix = np.round([np.random.uniform(-0.001, 0.001, [1, 785])], decimals=6)
        self.learn_count = 0
        self.activation_function = "Symmetrical Hard limit"
        self.learning_method = 'Filtered Learning'
        self.file_names = os.listdir("Data")
        self.learn_file_names = np.random.choice(self.file_names, 800, replace=False)
        self.test_file_names = np.setdiff1d(self.file_names, self.learn_file_names)
        self.train_set = self.generate_train_set()
        self.test_set = self.generate_test_set()
        self.train_target_vector = self.get_train_target()
        self.test_target_vector = self.get_test_target()
        self.test_output_vector = np.zeros((10, 200))
        self.train_output_vector = np.zeros((10, 800))
        self.eper = 0
        self.prev_error = 0
        self.error_matrix = [self.prev_error,self.eper]
        self.count_array = [self.learn_count-1, self.learn_count]




        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error Percentage')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        # settting learning rate slider
        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Learning Rate",
                                            command=lambda event: self.learning_rate_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_callback())
        self.learning_rate_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #setting Learn button

        self.Learn = tk.Button(self.sliders_frame, text="Learn", command=self.Learn_the_neuron)
        self.Learn.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #setting Randomize weights button

        self.Randomize_weights = tk.Button(self.sliders_frame, text="Randomize Weights", command=self.get_random_weights)
        self.Randomize_weights.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)


        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(6, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit","Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard limit")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Dropdown for Learning Method

        self.label_for_Learning_method = tk.Label(self.buttons_frame, text="Larning Method",
                                                      justify="center")
        self.label_for_Learning_method.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.Learning_method_variable = tk.StringVar()
        self.Learning_method_variable_dropdown = tk.OptionMenu(self.buttons_frame, self.Learning_method_variable,
                                                          "Filtered Learning", "Delta Rule", "Unsupervised Hebb",
                                                          command=lambda
                                                              event: self.learning_method_dropdown_callback())
        self.Learning_method_variable.set("Filtered Learning")
        self.Learning_method_variable_dropdown.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #self.display_activation_function()

        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    def net_value_callback(self, w, i):

        self.net_value = np.zeros((10, 200), dtype=np.float32)
        self.net_value[:] = np.dot(w, i)


        '''if self.activation_function == 'Symmetrical Hard limit':
           for i in range(0, 10):
              for j in range(0, 200):
                 if net_value[i, j] > 0:

                    #activation = 1.0
                    self.test_output_vector[i, j] = 1
                 else:
                    #activation = -1.0
                    self.test_output_vector[i, j] = 0
                #   #o[i, j] = activation


        elif self.activation_function == 'Linear':
          for i in range(0, 10):
             for j in range(0, 200):
                if net_value[i, j] < 1000:
                   self.test_output_vector[i, j] = np.float32(net_value[i, j])
                else:
                   self.test_output_vector[i, j] = 1000
                #    #o[i, j] = activation


        elif self.activation_function == 'Hyperbolic Tangent':
            for i in range(0, 10):
                for j in range(0, 200):
                    activation = (exp(net_value[i, j]) - exp(-net_value[i, j]))/(exp(net_value[i, j]) + exp(-net_value[i, j]))
                    self.test_output_vector[i, j] = activation


        #return self.test_output_vector'''
        return self.net_value

    def Learn_the_neuron(self):

        for i in range(0, 100):
            self.learn_count = self.learn_count +1
            if self.learning_method == "Filtered Learning":
                gamma = 0.1
                self.weight_matrix = (1 - gamma) * self.weight_matrix + (self.learning_rate * np.dot(self.train_target_vector, np.transpose(self.train_set)))


            elif self.learning_method == "Delta Rule":
                error = self.caluculate_the_error()
                self.weight_matrix = self.weight_matrix + (self.learning_rate * np.dot(error, np.transpose(self.train_set)))

            elif self.learning_method == "Unsupervised Hebb":
                output = self.calculate_the_output()
                self.weight_matrix = self.weight_matrix + self.learning_rate * np.dot(output, np.transpose(self.train_set))


            self.net_value_callback(w=self.weight_matrix, i= self.test_set)
            self.prev_error = self.eper
            self.eper = np.array(self.plot_the_error(count=i))
            self.count_array = [self.learn_count - 1, self.learn_count]
            self.error_matrix = [self.prev_error,self.eper]
            self.display_activation_function(self.count_array, self.error_matrix)


        #self.display_activation_function()

    def plot_the_error(self,count):

        non_matching_coloumns = 0
        for i in range(0, 200):
            max_value = np.max(self.net_value[:, i])
            for j in range(0, 10):
                if self.net_value[j, i] == max_value:
                    self.net_value[j, i] = 1
                else:
                    self.net_value[j, i] = 0

        for t in range(0, 200):
            if np.array_equal(self.net_value[:, t], self.test_target_vector[:, t]) == True:
                non_matching_coloumns = non_matching_coloumns
            else:
                non_matching_coloumns = non_matching_coloumns + 1



        error_percentage = (non_matching_coloumns/2)
        #self.error_matrix[count] = error_percentage
        return error_percentage



    def display_activation_function(self,x, y):

        self.axes.plot(x, y, color='blue')
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.axes.xaxis.set_visible(True)
        self.axes.yaxis.set_visible(True)
        plt.title(self.learning_method + " " +self.activation_function)
        self.canvas.draw()


    def learning_rate_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        #self.display_activation_function()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        #self.display_activation_function()

    def learning_method_dropdown_callback(self):
        self.learning_method = self.Learning_method_variable.get()
        #self.display_activation_function()

    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread("Data/" + file_name).astype(np.float32)  # read image and convert to float
        return img.reshape(-1, 1).reshape(1,784)  # reshape to column vector and return it

    def generate_train_set(self):
        self.train_set = np.zeros((785, 800))
        for i in range(0, 800):
            self.train_set[0:784, i] = self.read_one_image_and_convert_to_vector(self.learn_file_names[i])
        self.train_set[784, :] = 1
        self.train_set = (self.train_set/255) - 0.5
        return self.train_set

    def generate_test_set(self):
        self.test_set = np.zeros((785, 200))
        for i in range(0, 200):
            self.test_set[0:784, i] = self.read_one_image_and_convert_to_vector(self.test_file_names[i])
        self.test_set[784, :] = 1
        self.test_set = (self.test_set/255) - 0.5
        return self.test_set

    def get_train_target(self):
        self.train_target = np.zeros((10, 800))
        for i in range(0, 800):
            str = self.learn_file_names[i]
            if str[0] == "0":
                self.train_target[0, i] = 1
            elif str[0] == "1":
                self.train_target[1, i] = 1
            elif str[0] == "2":
                self.train_target[2, i] = 1
            elif str[0] == "3":
                self.train_target[3, i] = 1
            elif str[0] == "4":
                self.train_target[4, i] = 1
            elif str[0] == "5":
                self.train_target[5, i] = 1
            elif str[0] == "6":
                self.train_target[6, i] = 1
            elif str[0] == "7":
                self.train_target[7, i] = 1
            elif str[0] == "8":
                self.train_target[8, i] = 1
            elif str[0] == "9":
                self.train_target[9, i] = 1
        return self.train_target

    def caluculate_the_error(self):
        train_net_value = np.zeros((10,800), dtype=np.float32)
        train_net_value[:] = np.dot(self.weight_matrix, self.train_set)
        if self.activation_function == 'Symmetrical Hard limit':
            for i in range(0, 10):
                for j in range(0, 800):
                    if train_net_value[i, j] > 0:
                        self.train_output_vector[i, j] = 1
                    else:
                        self.train_output_vector[i, j] = -1

        elif self.activation_function == 'Linear':
            for i in range(0, 10):
                for j in range(0, 800):
                    if train_net_value[i, j] < 1000:
                        self.train_output_vector[i, j] = np.float32(train_net_value[i, j])
                    else:
                        self.train_output_vector[i, j] = 1000
                        # o[i, j] = activation

        elif self.activation_function == 'Hyperbolic Tangent':
            for i in range(0, 10):
                for j in range(0, 800):
                    train_net_value[i, j] = train_net_value[i, j ]/1000000
                    activation = np.round((np.exp(train_net_value[i, j]) - np.exp(-train_net_value[i, j])) / (
                    np.exp(train_net_value[i, j]) + np.exp(-train_net_value[i, j])))
                    self.train_output_vector[i, j] = activation

        error = self.train_target_vector - self.train_output_vector

        return error

    def get_test_target(self):
        self.test_target = np.zeros((10, 200))
        for i in range(0, 200):
            str = self.test_file_names[i]
            if str[0] == "0":
                self.test_target[0, i] = 1
            elif str[0] == "1":
                self.test_target[1, i] = 1
            elif str[0] == "2":
                self.test_target[2, i] = 1
            elif str[0] == "3":
                self.test_target[3, i] = 1
            elif str[0] == "4":
                self.test_target[4, i] = 1
            elif str[0] == "5":
                self.test_target[5, i] = 1
            elif str[0] == "6":
                self.test_target[6, i] = 1
            elif str[0] == "7":
                self.test_target[7, i] = 1
            elif str[0] == "8":
                self.test_target[8, i] = 1
            elif str[0] == "9":
                self.test_target[9, i] = 1
        return self.test_target

    def get_random_weights(self):
        self.weight_matrix = np.round([np.random.uniform(-0.001, 0.001, [1, 785])], decimals=6)
        self.learn_count = 0
        self.axes.cla()
        self.axes.cla()
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error Percentage')
        self.axes.set_title(self.learning_method +" "+ self.activation_function)
        self.canvas.draw()



    def test_the_weights(self):
        self.net_value_callback(w=self.weight_matrix,i=self.test_set)

    def calculate_the_output(self):
        train_net_value = np.zeros((10, 800), dtype=np.float32)
        train_net_value[:] = np.dot(self.weight_matrix, self.train_set)
        if self.activation_function == 'Symmetrical Hard limit':
            for i in range(0, 10):
                for j in range(0, 800):
                    if train_net_value[i, j] > 0:
                        self.train_output_vector[i, j] = 1
                    else:
                        self.train_output_vector[i, j] = -1

        elif self.activation_function == 'Linear':
            for i in range(0, 10):
                for j in range(0, 800):
                    if train_net_value[i, j] < 1000:
                        self.train_output_vector[i, j] = np.float32(train_net_value[i, j])
                    else:
                        self.train_output_vector[i, j] = 1000
                        # o[i, j] = activation

        elif self.activation_function == 'Hyperbolic Tangent':
            for i in range(0, 10):
                for j in range(0, 800):
                    train_net_value[i, j] = train_net_value[i, j]/1000000
                    activation = np.round((np.exp(train_net_value[i, j]) - np.exp(-train_net_value[i, j])) / (
                        np.exp(train_net_value[i, j]) + np.exp(-train_net_value[i, j])))
                    self.train_output_vector[i, j] = activation


        return self.train_output_vector





