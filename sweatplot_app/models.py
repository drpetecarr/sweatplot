from django.db.models import Model, CharField, ForeignKey, IntegerField, DateTimeField, CASCADE, Manager, FileField, FilePathField
from django.contrib import admin
import os
import pandas as pd
import numpy as np
from typing import List


class SessionAdmin(admin.ModelAdmin):
    """
    After save is pressed in admin view, the Session needs to read in csv file
    and save as a pd.DataFrame variable.
    """
    def save_model(self, request, obj, form, change):
        obj.save()


class SessionManager(Manager):
    pass


class PatientManager(Manager):
    pass


class Patient(Model):
    name = CharField(max_length=50)
    age = CharField(default='0', max_length=20)
    diagnosis = CharField(max_length=300)
    gender = CharField(choices=[('1', 'Female'), ('2', 'Male'), ('3', 'Other')], max_length=10)
    objects = PatientManager()

    def __str__(self) -> str:
        return self.name


def measure(func: classmethod) -> classmethod:
    """
    Wrapper that gives function an attribute 'measure' = True.
    This is to signal whether a function is returning some sort of statistical
    measure of the session.
    These functions can only have self as an argument.
    """

    def add_then_call(*args, **kwargs):
        func.measure = True
        return func(*args, **kwargs)

    return add_then_call


def additive(func: classmethod) -> classmethod:
    """
    Wrapper that gives function an attribute 'additive' = True.
    This is to signal whether a measure functions returned value will be valid,
    not just over one sessions data, but multiple sessions data concatenated.
    """

    def add_then_call(*args, **kwargs):
        func.additive = True
        return func(*args, **kwargs)

    return add_then_call


def clean_df(func: classmethod) -> classmethod:
    """ Wrapper for cleaning up data."""

    def clean_then_call(*args, **kwargs):
        assert args[0].df() is not None
        assert len(args[0].df()) > 10  # make sure data isn't too small
        return func(*args, **kwargs)

    return clean_then_call


class Session(Model):
    patient = ForeignKey(Patient, on_delete=CASCADE)
    number = IntegerField(default=0)
    datetime = DateTimeField('date of session')
    csv = FileField('full path to csv', max_length=200)
    frequency_bands = CharField('frequency bands', max_length=200, default='0.15, 0.5, 1.3, 1.6, 2.4')
    phase_bands = CharField('phase_bands', max_length=200,
                            default='0.00, 0.313, 0.615, 0.917, 1.211, 1.505, 1.800, 2.093')
    objects = SessionManager()
    _df = None

    def __str__(self) -> str:
        return 'PATIENT: ' + self.patient.name + ' SESSION NUMBER: ' + str(self.number)

    def measure_dictionary(self) -> dict:
        """
        Calls every method with an attribute 'measure' == True and stores result in a dictionary.
        These methods need to be without any arguments (other than self).
        :return: dict
        """
        measures = dict()
        method_list = []
        for func_str in dir(self):
            if not func_str == 'objects' and not func_str.startswith("__") and not\
                    func_str.startswith('measure_dictionary') and callable(getattr(self, func_str)):
                method_list.append(func_str)
        for func_str in method_list:
            func = getattr(self, func_str)
            try:
                result = func()
                print(func, file=open('dump.txt', 'w'))
            except:
                continue
            if hasattr(func, 'measure'):
                if func.measure:
                    print(func_str, file=open('dump.txt', 'w'))
                    measures[func_str] = result
        return measures

    def df(self, trim_start_by=2):
        """
        Loads in csv file as a pd.DataFrame. Knocks of the first couple of rows.
        """
        if self._df is None:
            self._df = pd.read_csv(os.path.join(os.getcwd(), self.csv.name))[trim_start_by:].reset_index()
        return self._df


    @clean_df
    @measure
    @additive
    def duration(self) -> int:
        return len(self.df()['Time'])

    @clean_df
    def right_hand_data(self) -> pd.Series:
        return self.df()['Right']

    @clean_df
    def left_hand_data(self) -> pd.Series:
        return self.df()['Left']

    @clean_df
    def frequency_data(self) -> pd.Series:
        return self.df()['Frequency']

    @clean_df
    def phase_data(self) -> pd.Series:
        return self.df()['Phase']

    @clean_df
    @measure
    @additive
    def convergence_magnitudes(self) -> List[float]:
        """
        Returns list size of df length with 0 for every non-convergent value
        and with the magnitude of the convergence for every convergent value.
        :return: list[float]
        """
        magnitudes = [0] * len(self.right_hand_data())
        for i in range(len(self.right_hand_data())):
            if i == 0:
                continue
            top = self.left_hand_data().copy()
            bot = self.right_hand_data().copy()
            if top[i] < bot[i]:  # then swap them round
                top, bot = bot, top
            if top[i] < top[i - 1]:  # top is decreasing
                if bot[i] > bot[i - 1]:  # bot is increasing. Therefore they're converging
                    size_of_convergence = top[i - 1] - top[i] + bot[i] - bot[i - 1]
                    magnitudes[i] = size_of_convergence
        return magnitudes

    @clean_df
    @measure
    @additive
    def divergence_magnitudes(self) -> List[float]:
        """
        Returns List size of df length with 0 for every non-divergent value
        and with the magnitude of the divergence for every divergent value.
        :return: list[float]
        """
        magnitudes = [0] * len(self.right_hand_data())
        for i in range(len(self.right_hand_data())):
            if i == 0:
                continue
            top = self.left_hand_data().copy()
            bot = self.right_hand_data().copy()
            if top[i] < bot[i]:  # then swap them round
                top, bot = bot, top
            if top[i] > top[i - 1]:  # top is increasing
                if bot[i] < bot[i - 1]:  # bot is decreasing. Therefore they're diverging
                    size_of_divergence = top[i] - top[i - 1] + bot[i - 1] - bot[i]
                    magnitudes[i] = size_of_divergence
        return magnitudes

    @clean_df
    def convergence_count_bagged_against_time(self, number_of_bags: int) -> List[int]:
        """
        Returns count of convergence in each time interval.
        :param number_of_bags: int
        :return: List[int]
        """
        assert number_of_bags != 0
        bags = [0] * number_of_bags
        for i, value in enumerate(self.convergence_magnitudes()):
            if value != 0:
                bag_pos = int((i / len(self.convergence_magnitudes())) * number_of_bags)
                bags[bag_pos] += 1
        return bags

    @clean_df
    def divergence_count_bagged_against_time(self, number_of_bags: int) -> List[int]:
        """
        Returns count of divergence in each time interval.
        :param number_of_bags: int
        :return: List[int]
        """
        assert number_of_bags != 0
        bags = [0] * number_of_bags
        for i, value in enumerate(self.divergence_magnitudes()):
            if value != 0:
                bag_pos = int((i / len(self.divergence_magnitudes())) * number_of_bags)
                bags[bag_pos] += 1
        return bags

    @clean_df
    @measure
    @additive
    def variance_convergence_magnitudes(self) -> float:
        """
        Returns variance of convergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        non_zero_magnitudes = [mag for mag in self.convergence_magnitudes() if mag != 0]
        return np.var(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    def variance_divergence_magnitudes(self) -> float:
        """
        Returns variance of divergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        non_zero_magnitudes = [mag for mag in self.divergence_magnitudes() if mag != 0]
        return np.var(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    def median_convergence_magnitude(self) -> float:
        """
        Returns median of convergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        non_zero_magnitudes = [mag for mag in self.convergence_magnitudes() if mag != 0]
        return np.median(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    def median_divergence_magnitude(self) -> float:
        """
        Returns median of divergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        non_zero_magnitudes = [mag for mag in self.divergence_magnitudes() if mag != 0]
        return np.median(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    def mean_divergence_magnitude(self) -> float:
        """
        Returns mean of divergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        non_zero_magnitudes = [mag for mag in self.divergence_magnitudes() if mag != 0]
        return np.mean(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    def mean_convergence_magnitude(self) -> float:
        """
        Returns mean of convergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        non_zero_magnitudes = [mag for mag in self.convergence_magnitudes() if mag != 0]
        return np.mean(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    def convergence_score(self) -> float:
        """
        Return the percentage of convergence instances. That is, when
        left and right hand values are both moving towards each other.
        :return: float
        """
        count = len([mag for mag in self.convergence_magnitudes() if mag != 0])
        return (count / len(self.left_hand_data())) * 100

    @clean_df
    @measure
    @additive
    def divergence_score(self) -> float:
        """
        Return the percentage of divergence instances. That is, when
        left and right hand values are both moving away from each other.
        :return: float
        """
        count = len([mag for mag in self.divergence_magnitudes() if mag != 0])
        return (count / len(self.left_hand_data())) * 100

    @clean_df
    @measure
    def gap_shift(self) -> float:
        """
        Returns the  in the shift between left hand and right hand
        from first reading to last reading. Right hand is assumed to be on top.
        :return: float
        """
        return self.right_shift - self.left_shift

    @clean_df
    @measure
    def left_shift(self) -> float:
        """
        Returns shift in left hand data from first value to last.
        Positive value means it's shifted upwards.
        :return: float
        """
        return self.left_hand_data().iloc[-1] - self.left_hand_data().iloc[0]

    @clean_df
    @measure
    def right_shift(self) -> float:
        """
        Returns shift in right hand data from first value to last.
        Positive value means it's shifted upwards.
        :return: float
        """
        return self.right_hand_data().iloc[-1] - self.right_hand_data().iloc[0]

    @clean_df
    def bag_values_by_bands(self, bands: List[float], column_header: str) -> List[List[float]]:
        """
        For a column in df specified by column_header, values are bagged according to bands.
        :param bands: List[float] must be sorted
        :param column_header: str must match a column header in df
        :return: List[int] of length len(bands) + 1
        """
        bags = [[]] * (len(bands) + 1)
        for value in self.df()[column_header]:
            for i, band in enumerate(bands):
                if value < band:  # if value is under current band, add value to bag
                    bags[i] += [value]  # after bags is updated, move on to next value
                    break
            bags[-1] += [value]  # if this line is reached then the value must be larger than highest band value,
            # so add it to last bag
        return bags

    @clean_df
    def count_bags(self, bags: List[List[float]]) -> List[int]:
        """
        Returns List of same size as bags but it's values are the size of each bag.
        :param bags: List[Lists of floats]
        :return: List[int]
        """
        return [len(bag) for bag in bags]


