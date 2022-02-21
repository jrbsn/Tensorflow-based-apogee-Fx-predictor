import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tkinter as tk
import time

class Inflation():
    def __init__(self):
        self.airData = pd.read_csv('airData.csv')
        self.delta_t = .02
        self.sims = 50000

        # placeholder to start dataset
        self.data = np.array([[0, 0, 0, 0, 0, 0, 0]])
    def main(self, input, training):
        if training == True:
            for i in range(self.sims):
                inputs = self.inputs()
                fx = self.simulation(inputs)
                self.append(inputs, fx)
                if i % 1000 == 0:
                    print("Simulation %i" % i)
        else:
            return self.simulation(input)
    def inputs(self):
        diameter = random.randint(6, 30)
        cd = random.choice([2.2, .97])
        weight = random.randint(10, 250)
        vls = random.randint(60, 180)
        aird = random.randint(1, 23) / 10000
        cfc = random.choice([8, 11.7])

        return diameter, cd, weight, vls, aird, cfc
    def simulation(self, inputs):
        diameter, cd, weight = inputs[0], inputs[1], inputs[2]
        vls, aird, cfc = inputs[3], inputs[4], inputs[5]

        area = (((diameter) / 2) ** 2) * 3.1415926
        cds = area * cd
        tf = (cfc * diameter) / vls

        tInt = self.delta_t
        iterations = int(tf / tInt)

        time = 0
        velocity = -1 * vls
        acceleration = 0
        drag = weight
        mass = weight / 32.172
        fx = 0

        for i in range(iterations):
            time = time + tInt
            acceleration = (drag - weight) / mass
            velocity = velocity + acceleration * tInt
            if ((.5 * aird * velocity ** 2) * cds * (time / tf) ** 2) < weight:
                drag = weight
            else:
                drag = (.5 * aird * velocity ** 2) * cds * (time / tf) ** 2

            if drag > fx:
                fx = int(drag)

        return fx
    def append(self, inputs, fx):
        input1 = list(inputs)
        input1.append(fx)
        newData = np.array([input1])
        self.data = np.append(self.data, newData, axis=0)
class RocketFlight():
    def __init__(self):
        self.airData = pd.read_csv('airData.csv')

        self.delta_t = .05
        self.sims = 25000

        # placeholder to start dataset
        self.data = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
        self.clears = 0
    def main(self, input, training):
        if training == True:
            past = (time.perf_counter())
            for i in range(self.sims):
                inputs = self.inputs()
                apogee = self.simulation(inputs)
                self.append(inputs, apogee)
                if i % 1000 == 0:
                    time.sleep(600)
                if i % 25 == 0:
                    new = (time.perf_counter())
                    rate = 25/(new - past)
                    past = new

                    print("Simulation Count: %i   Simulation Rate: %f sims/second" % (i, rate))

        else:
            return self.simulation(input)
    def inputs(self):
        diameter = random.randint(3, 10)  #Inches
        cd = random.randint(40, 80) / 100
        propFrac = random.randint(1, 6) / 10
        initAlt = random.randint(0, 10000)
        weight = random.randint(6,1000)
        burnTime = random.randint(3,25)
        avgThrust = random.randint(200,10000)  #Newtons


        if avgThrust < weight:
            self.inputs()

        return diameter, cd, weight, initAlt, burnTime, avgThrust, propFrac
    def simulation(self, inputs):

        diameter, cd, weight = (inputs[0] / 12), inputs[1], inputs[2]
        initAlt, burnTime, avgThrust, propFrac = inputs[3], inputs[4], (inputs[5] * 0.22480894244319), inputs[6]

        area = (((diameter) / 2) ** 2) * 3.1415926

        tInt = self.delta_t

        time = 0
        velocity = 0
        acceleration = 0
        drag = 0
        mass = weight / 32.172
        propMass = weight * propFrac
        flowRate = propMass * (tInt/burnTime)
        altitude = initAlt
        apogee = initAlt
        airD = float(self.airData[self.airData["Altitude"] > initAlt].iloc[0]["Density"])

        while velocity >= 0:

            if altitude > 190000:
                airD = 4.95E-07
            else:
                airD = float(self.airData[self.airData["Altitude"] > altitude].iloc[0]["Density"])

            time = time + tInt
            if time <= burnTime:
                thrust = avgThrust
                weight = weight - flowRate
            else:
                thrust = 0
            velocity = velocity + acceleration * tInt
            acceleration = (thrust - weight - drag) / mass
            drag = (.5 * airD * velocity ** 2) * cd * area
            altitude = altitude + (velocity * tInt + .5 * acceleration * tInt**2)

            if altitude > apogee:
                apogee = altitude - initAlt

        return apogee
    def append(self, inputs, apogee):
        input1 = list(inputs)
        input1.append(apogee)
        newData = np.array([input1])
        self.data = np.append(self.data, newData, axis=0)

        while self.data.shape[0] % 1000 == 0:
            print("Data Saved")
            self.clears = self.clears + 1
            dataframe = pd.DataFrame(self.data)
            dataframe.to_csv(r'C:\Users\josha\CSV\export_dataframe%s.csv' % self.clears)
            self.data = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
class NeuralNetwork():
    def __init__(self):
        pass
    def inflationNN(self, rawData):
        columnNames = ['Diameter', 'Cd', 'Weight', 'Vls', 'Aird', 'Cfc', 'Fx']
        rawDataset = pd.DataFrame(data=rawData, columns=columnNames)
        dataset = rawDataset.copy()
        dataset.tail()

        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('Fx')
        test_labels = test_features.pop('Fx')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        def build_and_compile_model(norm):
            model = keras.Sequential([
                norm,
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(loss='mean_absolute_error',
                          optimizer=tf.keras.optimizers.Adam(0.0001))
            return model

        dnn_model = build_and_compile_model(normalizer)
        dnn_model.summary()

        history = dnn_model.fit(
            train_features,
            train_labels,
            validation_split=0.2,
            verbose=2, epochs=1500)

        dnn_model.save('test_model2')
    def rocketNN(self, rawData):

        columnNames = ['Diameter', 'Cd', 'Weight', 'InitAlt', 'BurnTime', 'AvgThrust', 'PropFrac', 'Apogee']
        rawDataset = pd.DataFrame(data=rawData, columns=columnNames)
        dataset = rawDataset.copy()
        dataset.tail()

        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('Apogee')
        test_labels = test_features.pop('Apogee')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        def build_and_compile_model(norm):
            model = keras.Sequential([
                norm,
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(loss='mean_absolute_error',
                          optimizer=tf.keras.optimizers.Adam(0.0001))
            return model

        dnn_model = build_and_compile_model(normalizer)
        dnn_model.summary()

        history = dnn_model.fit(
             train_features,
             train_labels,
             validation_split=0.2,
             verbose=2, epochs=1500)

        dnn_model.save('rocket')
class GUI():
    def __init__(self):
        self.windowSize = '700x450'
        self.introFont = ("Montserrat", 46, "bold")
        self.labelFont = ("Montserrat", 16)
        self.buttonFont = ("Montserrat", 20)
        self.headingFont = ("Montserrat", 30, "bold")
        self.backgroundColor = '#%02x%02x%02x' % (6, 14, 25)
        self.otherColor = '#%02x%02x%02x' % (245, 225, 211)
    def inflationGUI(self, currentWindow):

        self.loaded_model = tf.keras.models.load_model('test_model2')
        currentWindow.destroy()
        self.inflationWindow = tk.Tk()
        main = self.inflationWindow
        main.geometry(self.windowSize)
        main.configure(bg=self.backgroundColor)

        def goToFlight():
            self.flightGUI(main)

        self.fxVar = tk.StringVar()
        self.fxVar2 = tk.StringVar()

        self.entries = ['diaInput', 'cdInput', 'weightInput', 'vlsInput', 'airdInput', 'cfcInput']
        labels = ["Diameter", "Drag Coefficient", "Weight", "Line Stretch Vel.", "Air Density", "Canopy Fill Constant"]

        for i in range(len(self.entries)):
            self.entries[i] = tk.Entry(main, font=self.labelFont, width=10)
            labels[i] = tk.Label(main, text=labels[i], fg='black', font=self.labelFont, bg=self.backgroundColor, width=20, anchor='center')

            self.entries[i].place(relx=.35, rely=.25+(i/8))
            labels[i].place(relx=0, rely=.25+(i/8))

        title = tk.Label(main, text="Opening Force Predictor", font=self.headingFont, anchor='center', fg='white',
                         bg="black", relief="ridge", borderwidth=4)
        title.place(relx=.2, rely=.075)

        button = tk.Button(main, text="ML Estimate", command=self.inflationML, font=self.labelFont)
        button.place(relx=.6, rely=.7)

        button3 = tk.Button(main, text="Sim. Estimate", command=self.inflationSim, font=self.labelFont)
        button3.place(relx=.6, rely=.45)

        button2 = tk.Button(main, text="Flight Predictor", command=goToFlight)
        button2.place(relx=.85,rely=.05)

        output = tk.Label(main, textvariable=self.fxVar, font=self.buttonFont, bg=self.backgroundColor, fg='white')
        output.place(relx=.85,rely=.7)

        output2 = tk.Label(main, textvariable=self.fxVar2, font=self.buttonFont, bg=self.backgroundColor, fg='white')
        output2.place(relx=.85, rely=.45)

        main.mainloop()
    def flightGUI(self, currentWindow):
        self.loaded_model = tf.keras.models.load_model('rocket')
        currentWindow.destroy()
        self.flightWindow = tk.Tk()
        main = self.flightWindow
        main.geometry(self.windowSize)
        main.configure(bg=self.backgroundColor)

        def goToFlight():
            self.flightGUI(main)

        self.apogeeML = tk.StringVar()
        self.apogeeSIM = tk.StringVar()

        self.entries = ['dia', 'cd', 'weight', 'initAlt', 'bt', 'avgThrust', 'propF']
        labels = ["Diameter", "Drag Coefficient", "Weight", "Launch Altitude", "Burn Time", "Avg. Thrust", "Prop Fraction"]

        for i in range(len(self.entries)):
            self.entries[i] = tk.Entry(main, font=self.labelFont, width=10)
            labels[i] = tk.Label(main, text=labels[i], fg='white', font=self.labelFont, bg=self.backgroundColor,
                                 width=20, anchor='center')

            self.entries[i].place(relx=.35, rely=.25 + (i / 10))
            labels[i].place(relx=0, rely=.25 + (i / 10))

        title = tk.Label(main, text="Apogee Predictor", font=self.headingFont, anchor='center', fg='white',
                         bg=self.backgroundColor)
        title.place(relx=.25, rely=.075)

        button = tk.Button(main, text="ML Estimate", command=self.flightML, font=self.labelFont)
        button.place(relx=.6, rely=.7)

        button3 = tk.Button(main, text="Sim. Estimate", command=self.flightSim, font=self.labelFont)
        button3.place(relx=.6, rely=.45)

        button2 = tk.Button(main, text="Flight Predictor", command=goToFlight)
        button2.place(relx=.85, rely=.05)

        output = tk.Label(main, textvariable=self.apogeeML, font=self.buttonFont, bg=self.backgroundColor, fg='white')
        output.place(relx=.85, rely=.7)

        output2 = tk.Label(main, textvariable=self.apogeeSIM, font=self.buttonFont, bg=self.backgroundColor, fg='white')
        output2.place(relx=.85, rely=.45)
    def centerWindow(self, window):
        pass
    def inflationML(self):
        data = ['dia', 'cd', 'weight', 'vls', 'aird', 'cfc']

        for i in range(len(data)):
            data[i] = float(self.entries[i].get())

        fx = int(self.loaded_model.predict(data))

        self.fxVar.set(str(fx))
        self.inflationWindow.update_idletasks()
    def inflationSim(self):
        data = ['dia', 'cd', 'weight', 'vls', 'aird', 'cfc']

        for i in range(len(data)):
            data[i] = float(self.entries[i].get())

        inf = Inflation()
        fx = inf.main(data, False)
        self.fxVar2.set(str(fx))
        self.inflationWindow.update_idletasks()
    def flightML(self):
        data = ['dia', 'cd', 'weight', 'initAlt', 'bt', 'avgThrust', 'propF']

        for i in range(len(data)):
            data[i] = float(self.entries[i].get())

        apogee = int(self.loaded_model.predict(data))

        self.apogeeML.set(str(apogee))
        self.flightWindow.update_idletasks()
    def flightSim(self):
        data = ['dia', 'cd', 'weight', 'initAlt', 'bt', 'avgThrust', 'propF']

        for i in range(len(data)):
            data[i] = float(self.entries[i].get())

        flight = RocketFlight()
        apogee = int(flight.main(data, False))

        self.apogeeSIM.set(str(apogee))
        self.flightWindow.update_idletasks()
    def startup(self):
        self.font = ("Montserrat", 0)

        self.startupWindow = tk.Tk()
        main = self.startupWindow
        main.geometry(self.windowSize)


        canvas = tk.Canvas(main, width=700, height=450)
        canvas.place(x=-2,y=-1)

        pic = tk.PhotoImage(file='intropic.png')
        canvas.create_image(0, 0,image=pic, anchor='nw')

        def goToInflation():
            (self.inflationGUI(self.startupWindow))

        def goToFlight():
            (self.flightGUI(self.startupWindow))

        def mainFun():
            title = tk.Label(self.startupWindow, text="ML Rocket Suite", font=self.introFont, bg="black", fg="white",
                             relief="ridge", borderwidth=3, anchor='center', padx=10)
            title.place(relx=.155, rely=.36)

            time.sleep(1)

            button1 = tk.Button(self.startupWindow, text="Fx Predictor", command=goToInflation,
                                bg="white", fg="black", relief="raised", borderwidth=5, font=self.buttonFont)
            button2 = tk.Button(self.startupWindow, text="Apogee Predictor", command=goToFlight,
                                bg="white", fg="black", relief="raised", borderwidth=5, font=self.buttonFont)
            button1.place(relx=.2, rely=.6)
            button2.place(relx=.5, rely=.6)

        mainFun()

        main.mainloop()

############################################################

GUI = GUI()
GUI.startup()