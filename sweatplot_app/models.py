import operator
import csv
from cycler import cycler
from django.db.models import Model, CharField, ForeignKey, IntegerField, DateTimeField, CASCADE, Manager, FileField, \
    FilePathField
from django.contrib import admin
import os
import pandas as pd
import numpy as np
from typing import List
import matplotlib

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# TODO overview graph on session page
# TODO live feed plotter integrated with PicoLogger
# TODO group patient option (must include option to group ALL patients as well as by age, diagnosis etc.
# TODO (this can come later))
# TODO sort out graphs!
# TODO sppeeeed up!
# TODO convergence/divergence count against phase/frequency


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


class MultipleSessions:
    def __init__(self, sessions: List):
        self.sessions = sessions

    def mean_convergence_magnitudes_total(self) -> float:
        """
        Average of sessions convergence_magnitudes_total
        :return: float
        """
        return round(sum([s.convergence_magnitudes_total() for s in self.sessions]) / len(self.sessions), 3)

    def mean_divergence_magnitudes_total(self) -> float:
        """
        Average of sessions divergence_magnitudes_total
        :return: float
        """
        return round(sum([s.divergence_magnitudes_total() for s in self.sessions]) / len(self.sessions), 3)

    def mean_convergence_magnitudes_split_total(self) -> List[float]:
        """
        Average of sessions convergence_magnitudes_split_total
        :return: list of 2 floats
        """
        left_mean = sum([s.convergence_magnitudes_split_total()[0] for s in self.sessions]) / len(self.sessions)
        right_mean = sum([s.convergence_magnitudes_split_total()[1] for s in self.sessions]) / len(self.sessions)
        return [round(left_mean, 3), round(right_mean, 3)]

    def mean_divergence_magnitudes_split_total(self) -> List[float]:
        """
        Average of sessions divergence_magnitudes_split_total
        :return: list of 2 floats
        """
        left_mean = sum([s.divergence_magnitudes_split_total()[0] for s in self.sessions]) / len(self.sessions)
        right_mean = sum([s.divergence_magnitudes_split_total()[1] for s in self.sessions]) / len(self.sessions)
        return [round(left_mean, 3), round(right_mean, 3)]

    def frequency_bands_by_session_number(self) -> List[tuple]:
        """
        Returns list of 2-tuples. First tuple value is a list of frequency bands, second value is a list of all
        sessions with those frequency bands.
        :return: list of 2-tuples
        """
        bands = []
        session_numbers = []
        for s in self.sessions:
            new_bands = s.frequency_bands.split(', ')
            for i, b in enumerate(bands):
                if new_bands == b:
                    session_numbers[i].append(s.number)
                    break
            else:
                bands.append(new_bands)
                session_numbers.append([s.number])
        assert (len(bands) == len(session_numbers))
        return [(bands[i], session_numbers[i]) for i in range(len(bands))]

    def phase_bands_by_session_number(self) -> List[tuple]:
        """
        Returns list of 2-tuples. First tuple value is a list of phase bands, second value is a list of all
        sessions with those phase bands.
        :return: list of 2-tuples
        """
        bands = []
        session_numbers = []
        for s in self.sessions:
            new_bands = s.phase_bands.split(', ')
            for i, b in enumerate(bands):
                if new_bands == b:
                    session_numbers[i].append(s.number)
                    break
            else:
                bands.append(new_bands)
                session_numbers.append([s.number])
        assert (len(bands) == len(session_numbers))
        return [(bands[i], session_numbers[i])
                or i in range(len(bands))]

    def convergence_magnitudes_against_time(self) -> List[float]:
        """
        Returns a list of 100 floats. Each float represents the mean of the convergence magnitudes that occurred at
        that corresponding percentage through all the sessions.
        :return: list of 100 floats
        """
        result = [0] * 100
        for s in self.sessions:
            convergence_magnitudes = s.convergence_magnitudes()
            convergence_magnitudes_squashed = squasher(convergence_magnitudes, 100)
            for i, val in enumerate(convergence_magnitudes_squashed):
                result[i] += val

        # take means
        for i in range(len(result)):
            result[i] /= len(self.sessions)

        return result

    def divergence_magnitudes_against_time(self) -> List[float]:
        """
        Returns a list of 100 floats. Each float represents the mean of the divergence magnitudes that occurred at
        that corresponding percentage through all the sessions.
        :return: list of 100 floats
        """
        result = [0] * 100
        for s in self.sessions:
            divergence_magnitudes = s.divergence_magnitudes()
            divergence_magnitudes_squashed = squasher(divergence_magnitudes, 100)
            for i, val in enumerate(divergence_magnitudes_squashed):
                result[i] += val

        # take means
        for i in range(len(result)):
            result[i] /= len(self.sessions)

        return result

    def convergence_magnitudes_by_hand_against_time(self) -> List[tuple]:
        """
        Returns a list of 2-tuples. First entry in tuple represents left hand, second tuple represents
        right hand. Each float represents the mean of the convergence magnitudes that occurred at that corresponding
        percentage through all the sessions for that hand.
        :return: list of 100 2-tuples
        """
        result = [(0, 0)] * 100
        for s in self.sessions:
            convergence_magnitudes_split = s.convergence_magnitudes_split()
            convergence_magnitudes_squashed = squasher(convergence_magnitudes_split, 100, tuples=True)
            for i, val in enumerate(convergence_magnitudes_squashed):
                result[i] = tuple(map(operator.add, result[i], val))

        # take means
        for i in range(len(result)):
            result[i] = (result[i][0] / len(self.sessions), result[i][1] / len(self.sessions))

        return result

    def divergence_magnitudes_by_hand_against_time(self) -> List[tuple]:
        """
        Returns a list of 2-tuples. First entry in tuple represents left hand, second tuple represents
        right hand. Each float represents the mean of the divergence magnitudes that occurred at that corresponding
        percentage through all the sessions for that hand.
        :return: list of 100 2-tuples
        """
        result = [(0, 0)] * 100
        for s in self.sessions:
            divergence_magnitudes_split = s.divergence_magnitudes_split()
            divergence_magnitudes_squashed = squasher(divergence_magnitudes_split, 100, tuples=True)
            for i, val in enumerate(divergence_magnitudes_squashed):
                result[i] = tuple(map(operator.add, result[i], val))

        # take means
        for i in range(len(result)):
            result[i] = (result[i][0] / len(self.sessions), result[i][1] / len(self.sessions))

        return result

    def convergence_magnitudes_bagged_by_phases(self) -> List[dict]:
        """
        Returns a list of dictionaries. Each dictionary corresponds to a different phase banding found in self.sessions.
        Each dict contains values for: phases, convergences at those phases, session numbers which have those phases.
        :return: List of dictionaries
        """
        result = []
        phase_bands_by_session_number = self.phase_bands_by_session_number()
        for bands, session_numbers in phase_bands_by_session_number:
            sum_convergences = [0] * (len(bands) + 1)
            for num in session_numbers:
                session = None
                for s in self.sessions:
                    if s.number == num:
                        session = s
                        break
                else:
                    Exception('session numbers don\'t match')
                convergences = session.convergence_magnitudes_bagged_by_phase()
                sum_convergences = [sum_convergences[i] + convergences[i] for i in range(len(sum_convergences))]
            d = dict()
            d['index'] = range(len(bands) + 1)
            d['bands'] = bands
            d['session_numbers'] = session_numbers
            d['result'] = [s / len(session_numbers) for s in sum_convergences]
            assert (len(d['index']) == len(d['result']))
            result.append(d)
        return result

    def divergence_magnitudes_bagged_by_phases(self) -> List[dict]:
        """
        Returns a list of dictionaries. Each dictionary corresponds to a different phase banding found in self.sessions.
        Each dict contains values for: phases, convergences at those phases, session numbers which have those phases.
        :return: List of dictionaries
        """
        result = []
        phase_bands_by_session_number = self.phase_bands_by_session_number()
        for bands, session_numbers in phase_bands_by_session_number:
            sum_divergences = [0] * (len(bands) + 1)
            for num in session_numbers:
                session = None
                for s in self.sessions:
                    if s.number == num:
                        session = s
                        break
                else:
                    Exception('session numbers don\'t match')
                divergences = session.divergence_magnitudes_bagged_by_phase()
                sum_divergences = [sum_divergences[i] + divergences[i] for i in range(len(sum_divergences))]
            d = dict()
            d['index'] = range(len(bands) + 1)
            d['bands'] = bands
            d['session_numbers'] = session_numbers
            d['result'] = [s / len(session_numbers) for s in sum_divergences]
            assert (len(d['index']) == len(d['result']))
            result.append(d)
        return result

    def convergence_magnitudes_bagged_by_frequencies(self) -> List[dict]:
        """
        Returns a list of dictionaries. Each dictionary corresponds to a different frequency banding in self.sessions.
        Each dict contains values for: frequencies, convergences at those frequencies, session numbers with frequencies.
        :return: List of dictionaries
        """
        result = []
        frequency_bands_by_session_number = self.frequency_bands_by_session_number()
        for bands, session_numbers in frequency_bands_by_session_number:
            sum_divergences = [0] * (len(bands) + 1)
            for num in session_numbers:
                session = None
                for s in self.sessions:
                    if s.number == num:
                        session = s
                        break
                else:
                    Exception('session numbers don\'t match')
                divergences = session.divergence_magnitudes_bagged_by_frequency()
                sum_divergences = [sum_divergences[i] + divergences[i] for i in range(len(sum_divergences))]
            d = dict()
            d['index'] = range(len(bands) + 1)
            d['bands'] = bands
            d['session_numbers'] = session_numbers
            d['result'] = [s / len(session_numbers) for s in sum_divergences]
            assert (len(d['index']) == len(d['result']))
            result.append(d)
        return result

    def divergence_magnitudes_bagged_by_frequencies(self) -> List[dict]:
        """
        Returns a list of dictionaries. Each dictionary corresponds to a different frequency banding found in
        self.sessions. Each dict contains values for: frequencies, divergences at those frequencies, session numbers
        which have those frequencies.
        :return: List of dictionaries
        """
        result = []
        frequency_bands_by_session_number = self.frequency_bands_by_session_number()
        for bands, session_numbers in frequency_bands_by_session_number:
            sum_divergences = [0] * (len(bands) + 1)
            for num in session_numbers:
                session = None
                for s in self.sessions:
                    if s.number == num:
                        session = s
                        break
                else:
                    Exception('session numbers don\'t match')
                divergences = session.divergence_magnitudes_bagged_by_frequency()
                sum_divergences = [sum_divergences[i] + divergences[i] for i in range(len(sum_divergences))]
            d = dict()
            d['index'] = range(len(bands) + 1)
            d['bands'] = bands
            d['session_numbers'] = session_numbers
            d['result'] = [s / len(session_numbers) for s in sum_divergences]
            assert (len(d['index']) == len(d['result']))
            result.append(d)
        return result


def squasher(to_be_squashed: List, squash_length: int, tuples=False) -> List:
    """
    Takes in a list of any length and squashes it to the length specified. Values that are squashed to
    the same location are summed.

    If tuples is True then the to_be_squashed list is assumed to compromise of tuples and their values are
    summed piecewise instead of appending.

    NOTE this can squash to a larger length, i.e., squash_length can be greater than len(to_be_squashed)
    :param tuples: bool
    :param to_be_squashed: list
    :type squash_length: int
    :return: list of length squash_length
    """
    if tuples:
        result = [tuple([0] * len(to_be_squashed[0]))] * squash_length
    else:
        result = [0] * squash_length
    for i, value in enumerate(to_be_squashed):
        ratio_through = i / len(to_be_squashed)
        new_position = int(squash_length * ratio_through)
        if tuples:
            result[new_position] = tuple(map(operator.add, result[new_position], value))
            continue
        result[new_position] += value
    return result


class Patient(Model):
    name = CharField(max_length=50)
    age = CharField(default='0', max_length=20)
    diagnosis = CharField(max_length=300)
    gender = CharField(choices=[('Female', 'Female'), ('Male', 'Male'), ('Other', 'Other')], max_length=10)
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

    add_then_call.__name__ = func.__name__
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

    add_then_call.__name__ = func.__name__
    return add_then_call


def round_by(func: classmethod, by=3) -> classmethod:
    """Rounds numbers to given number of digits"""

    def return_rounded(*args, **kwargs):
        return round(func(*args, **kwargs), by)

    return_rounded.__name__ = func.__name__
    return return_rounded


def clean_df(func: classmethod) -> classmethod:
    """ Wrapper for cleaning up data."""

    def clean_then_call(*args, **kwargs):
        assert args[0].df() is not None
        assert len(args[0].df()) > 10  # make sure data isn't too small
        return func(*args, **kwargs)

    clean_then_call.__name__ = func.__name__
    return clean_then_call


def multiple_to_csv(results: List, patient_name: str, measure_name: str):
    """
    Saves a csv of the result to a 'Patients' directory located in the working dir.
    measure_func can return either a value or an iterable.
    :param results: list of dicts
    :param patient_name: str
    :param measure_name: str
    """
    if not os.path.exists(os.path.join(os.getcwd(), 'Patients')):  # create a 'Patients' dir if it doesnt exist
        os.makedirs(os.path.join(os.getcwd(), 'Patients'))
    if not os.path.exists(os.path.join(os.getcwd(), 'Patients',
                                       patient_name)):  # create a ../Patients/patient_name dir if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), 'Patients', patient_name))
    for i, entry in enumerate(results):
        new_csv_path = os.path.join(os.getcwd(), 'Patients', patient_name, measure_name + str(i) + '.csv')
        with open(new_csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[key for key in entry])
            writer.writeheader()
            for key in entry:
                if hasattr(entry[key], '__iter__'):
                    writer.writerow({key: entry[key]})


def to_csv(result: iter, patient_name: str, measure_name: str):
    """
    Saves a csv of the result to a 'Patients' directory located in the working dir.
    measure_func can return either a value or an iterable.
    :param result: iterable
    :param patient_name: str
    :param measure_name: str
    """
    if not os.path.exists(os.path.join(os.getcwd(), 'Patients')):  # create a 'Patients' dir if it doesnt exist
        os.makedirs(os.path.join(os.getcwd(), 'Patients'))
    if not os.path.exists(os.path.join(os.getcwd(), 'Patients',
                                       patient_name)):  # create a ../Patients/patient_name dir if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), 'Patients', patient_name))
    new_csv_path = os.path.join(os.getcwd(), 'Patients', patient_name, measure_name + '.csv')
    with open(new_csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        try:
            [writer.writerow(row) for row in result]
        except:
            writer.writerow(result)


def create_bar_chart(result: List, index: List, patient: Patient, session, image_path='temp_plot.jpg'):
    """
    Saves a bar_chart of the measures output and returns the image path.
    :param patient: Patient
    :param session: Session
    :param result: list
    :param index: list
    :param image_path: str
    """
    plt.figure()
    plt.bar(index, result)
    plt.title(patient.name + '  (session: ' + str(session.number) + ')')
    plt.savefig(os.path.join(os.getcwd(), 'sweatplot_app', 'static', 'sweatplot_app', image_path))
    plt.close()


def create_multiple_bar_charts(results: List[dict], image_paths: List[str], patient: Patient):
    """
    Saves multiple bar chart images in the locations specified in image_paths.
    :param patient: Patient
    :param results: list of dicts
    :param image_paths: list of strings
    """
    assert (len(results) == len(image_paths))
    for i, d in enumerate(results):
        plt.figure()
        result = d['result']
        index = d['index']
        plt.bar(index, result)
        try:
            plt.title(patient.name + '  (bands: ' + str(d['bands']) + ')')
        except KeyError:
            plt.title(patient.name)
        plt.savefig(os.path.join(os.getcwd(), 'sweatplot_app', 'static', 'sweatplot_app', image_paths[i]))
        plt.close()


def create_graph(result, patient: Patient, session, image_path='temp_plot.jpg'):
    """
    Saves a graph of the measures output and returns the image path.

    :param patient: Patient
    :param session: Session
    :param result: iterable
    :param image_path: str
    """
    plt.figure()
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b'])))
    plt.title(patient.name + '  (session number: ' + str(session.number) + ')')
    left = list(session.left_hand_data())
    max_hand = max(list(map(np.abs, left)))
    right = list(session.right_hand_data())
    max_hand = max(list(map(np.abs, right)) + [max_hand])
    max_result = max(list(np.array(result).flatten()))
    left = [(val / max_hand) * max_result for val in left]
    right = [val / max_hand * max_result for val in right]
    left_right = list(zip(left, right))
    assert (len(left_right) == len(result))
    plt.plot(result, alpha=0.8)
    plt.plot(left_right, alpha=0.4)
    plt.savefig(os.path.join(os.getcwd(), 'sweatplot_app', 'static', 'sweatplot_app', image_path))
    plt.close()


def create_all_sessions_graph(results, patient: Patient, image_path='temp_plot.jpg'):
    """
    Saves a graph of the results.

    :param patient: Patient
    :param results: iterable
    :param image_path: str
    """
    plt.figure()
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b'])))
    plt.title(patient.name)
    plt.plot(results, alpha=0.8)
    plt.savefig(os.path.join(os.getcwd(), 'sweatplot_app', 'static', 'sweatplot_app', image_path))
    plt.close()


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
        return 'Session  ' + str(self.number)

    def convergence_magnitudes_split_total(self) -> List[float]:
        """
        Returns 2-list with left/right hand total convergence magnitudes.
        :return: 2-list
        """
        sum_of_mags = [0, 0]
        for tup in self.convergence_magnitudes_split():
            sum_of_mags[0] += tup[0]
            sum_of_mags[1] += tup[1]
        return [round(sum_of_mags[0], 3), round(sum_of_mags[1], 3)]

    def divergence_magnitudes_split_total(self) -> List[float]:
        """
        Returns 2-list with left/right hand total divergence magnitudes.
        :return: 2-list
        """
        sum_of_mags = [0, 0]
        for tup in self.divergence_magnitudes_split():
            sum_of_mags[0] += tup[0]
            sum_of_mags[1] += tup[1]
        return [round(sum_of_mags[0], 3), round(sum_of_mags[1], 3)]

    @round_by
    def convergence_magnitudes_total(self) -> float:
        """
        Returns float of left hand convergence + right hand convergence.
        :return: float
        """
        sum_of_mags = abs(self.convergence_magnitudes_split_total()[0]) + abs(
            self.convergence_magnitudes_split_total()[1])
        return sum_of_mags

    @round_by
    def divergence_magnitudes_total(self) -> float:
        """
        Returns float of left hand divergence + right hand divergence.
        :return: float
        """
        sum_of_mags = abs(self.divergence_magnitudes_split_total()[0]) + abs(
            self.divergence_magnitudes_split_total()[1])
        return sum_of_mags

    def measure_dictionary(self) -> dict:
        """
        Calls every method with an attribute 'measure' == True and stores result in a dictionary.
        These methods need to be without any arguments (other than self).
        :return: dict
        """
        measures = dict()
        method_list = []
        for func_str in dir(self):
            if not func_str == 'objects' and not func_str.startswith("__") and not \
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
        left_hand_data = self.left_hand_data()
        right_hand_data = self.right_hand_data()
        magnitudes = [0] * len(right_hand_data)
        for i in range(len(right_hand_data)):
            if i == 0:
                continue
            top = left_hand_data.copy()
            bot = right_hand_data.copy()
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
        left_hand_data = self.left_hand_data()
        right_hand_data = self.right_hand_data()
        magnitudes = [0] * len(right_hand_data)
        for i in range(len(right_hand_data)):
            if i == 0:
                continue
            top = left_hand_data.copy()
            bot = right_hand_data.copy()
            if top[i] < bot[i]:  # then swap them round
                top, bot = bot, top
            if top[i] > top[i - 1]:  # top is increasing
                if bot[i] < bot[i - 1]:  # bot is decreasing. Therefore they're diverging
                    size_of_divergence = top[i] - top[i - 1] + bot[i - 1] - bot[i]
                    magnitudes[i] = size_of_divergence
        return magnitudes

    @measure
    @additive
    @clean_df
    def divergence_magnitudes_split(self) -> List[tuple]:
        """
        Returns List with size of df length with (0,0) for every non-divergent value
        and with magnitude of left, right divergence respectively for divergent values.
        So if there is a divergence and left hand is top, then tuple will have positive
        first value and negative second value.
        :return: list(tuple(float, float))
        """
        left_hand_data = self.left_hand_data()
        right_hand_data = self.right_hand_data()
        magnitudes = [(0, 0)] * len(right_hand_data)
        for i in range(len(right_hand_data)):
            if i == 0:
                continue
            top = left_hand_data.copy()
            bot = right_hand_data.copy()
            left_top = True
            if top[i] < bot[i]:  # then swap them round
                left_top = False
                top, bot = bot, top
            if top[i] > top[i - 1]:  # top is increasing
                if bot[i] < bot[i - 1]:  # bot is decreasing. Therefore they're diverging
                    size_of_top_div = top[i] - top[i - 1]
                    size_of_bot_div = bot[i] - bot[i - 1]
                    if left_top:
                        magnitudes[i] = (size_of_top_div, size_of_bot_div)
                        continue
                    magnitudes[i] = (size_of_bot_div, size_of_top_div)

        return magnitudes

    @measure
    @additive
    @clean_df
    def convergence_magnitudes_split(self) -> List[tuple]:
        """
        Returns List with size of df length with (0,0) for every non-convergent value
        and with magnitude of left, right convergence respectively for convergent values.
        So if there is a convergence and left hand is top, then tuple will have positive
        first value and negative second value.
        :return: list(tuple(float, float))
        """
        left_hand_data = self.left_hand_data()
        right_hand_data = self.right_hand_data()
        magnitudes = [(0, 0)] * len(right_hand_data)
        for i in range(len(self.right_hand_data())):
            if i == 0:
                continue
            top = left_hand_data.copy()
            bot = right_hand_data.copy()
            left_top = True
            if top[i] < bot[i]:  # then swap them round
                left_top = False
                top, bot = bot, top
            if top[i] < top[i - 1]:  # top is decreasing
                if bot[i] > bot[i - 1]:  # bot is increasing. Therefore they're converging
                    size_of_top_con = top[i] - top[i - 1]
                    size_of_bot_con = bot[i] - bot[i - 1]
                    if left_top:
                        magnitudes[i] = (size_of_top_con, size_of_bot_con)
                        continue
                    magnitudes[i] = (size_of_bot_con, size_of_top_con)

        return magnitudes

    @clean_df
    def convergence_count_bagged_against_time(self, number_of_bags: int) -> List[int]:
        """
        Returns count of convergence in each time interval.
        :param number_of_bags: int
        :return: List[int]
        """
        assert number_of_bags != 0
        convergence_magnitudes = self.convergence_magnitudes()
        bags = [0] * number_of_bags
        for i, value in enumerate(convergence_magnitudes):
            if value != 0:
                bag_pos = int((i / len(convergence_magnitudes)) * number_of_bags)
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
        divergence_magnitudes = self.divergence_magnitudes()
        bags = [0] * number_of_bags
        for i, value in enumerate(divergence_magnitudes):
            if value != 0:
                bag_pos = int((i / len(divergence_magnitudes)) * number_of_bags)
                bags[bag_pos] += 1
        return bags

    # TODO redundant?
    @clean_df
    @measure
    def divergence_count_time_percentage(self) -> List[int]:
        """
        Returns count of divergence for each time step, where a time step is duration/100.
        :return: list[int]
        """
        return self.divergence_count_bagged_against_time(100)

    # TODO redundant?
    @clean_df
    @measure
    def convergence_count_time_percentage(self) -> List[int]:
        """
        Returns count of convergence for each time step, where a time step is duration/100.
        :return: list[int]
        """
        return self.convergence_count_bagged_against_time(100)

    @clean_df
    @measure
    @additive
    @round_by
    def variance_convergence_magnitudes(self) -> float:
        """
        Returns variance of convergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        convergence_magnitudes = self.convergence_magnitudes()
        non_zero_magnitudes = [mag for mag in convergence_magnitudes if mag != 0]
        return np.var(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    @round_by
    def variance_divergence_magnitudes(self) -> float:
        """
        Returns variance of divergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        divergence_magnitudes = self.divergence_magnitudes()
        non_zero_magnitudes = [mag for mag in divergence_magnitudes if mag != 0]
        return np.var(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    @round_by
    def median_convergence_magnitudes(self) -> float:
        """
        Returns median of convergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        convergence_magnitudes = self.convergence_magnitudes()
        non_zero_magnitudes = [mag for mag in convergence_magnitudes if mag != 0]
        return np.median(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    @round_by
    def median_divergence_magnitudes(self) -> float:
        """
        Returns median of divergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        divergence_magnitudes = self.divergence_magnitudes()
        non_zero_magnitudes = [mag for mag in divergence_magnitudes if mag != 0]
        return np.median(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    @round_by
    def mean_divergence_magnitudes(self) -> float:
        """
        Returns mean of divergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        divergence_magnitudes = self.divergence_magnitudes()
        non_zero_magnitudes = [mag for mag in divergence_magnitudes if mag != 0]
        return np.mean(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    @round_by
    def mean_convergence_magnitudes(self) -> float:
        """
        Returns mean of convergence magnitudes EXCLUDING 0 values in calculation.
        :return: float
        """
        convergence_magnitudes = self.convergence_magnitudes()
        non_zero_magnitudes = [mag for mag in convergence_magnitudes if mag != 0]
        return np.mean(non_zero_magnitudes)

    @clean_df
    @measure
    @additive
    @round_by
    def convergence_score(self) -> float:
        """
        Return the percentage of convergence instances. That is, when
        left and right hand values are both moving towards each other.
        :return: float
        """
        convergence_magnitudes = self.convergence_magnitudes()
        count = len([mag for mag in convergence_magnitudes if mag != 0])
        return (count / len(self.left_hand_data())) * 100

    @clean_df
    @measure
    @additive
    @round_by
    def divergence_score(self) -> float:
        """
        Return the percentage of divergence instances. That is, when
        left and right hand values are both moving away from each other.
        :return: float
        """
        divergence_magnitudes = self.divergence_magnitudes()
        count = len([mag for mag in divergence_magnitudes if mag != 0])
        return (count / len(self.left_hand_data())) * 100

    @clean_df
    @measure
    @round_by
    def total_shift(self) -> float:
        """
        Returns the  in the shift between left hand and right hand
        from first reading to last reading. Right hand is assumed to be on top.
        :return: float
        """
        return self.right_shift() - self.left_shift()

    @clean_df
    @measure
    @round_by
    def left_shift(self) -> float:
        """
        Returns shift in left hand data from first value to last.
        Positive value means it's shifted upwards.
        :return: float
        """
        return self.left_hand_data().iloc[-1] - self.left_hand_data().iloc[0]

    @clean_df
    @measure
    @round_by
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
        df = self.df()
        for value in df[column_header]:
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

    @clean_df
    @measure
    def convergence_magnitudes_bagged_by_phase(self) -> List[float]:
        """
        Returns a list of floats with length the same as self.phase_bands. Each float
        is the sum of all convergence magnitudes that occurred at that band.
        :return: List[float]
        """
        # the +1 is necessary since there is a bag above last threshold
        phase_bands = self.phase_bands.split(', ')
        magnitudes_bagged = [0] * (len(phase_bands) + 1)
        phase_data = self.phase_data()
        convergence_magnitudes = self.convergence_magnitudes()
        for i, mag in enumerate(convergence_magnitudes):
            if mag != 0:
                phase = phase_data[i]
                for j, band in enumerate(phase_bands):
                    if phase < band:
                        magnitudes_bagged[j] += mag
                        break
                else:
                    magnitudes_bagged[-1] += mag
        assert (len(magnitudes_bagged) == len(phase_bands) + 1)
        return magnitudes_bagged

    @clean_df
    @measure
    def convergence_magnitudes_bagged_by_frequency(self) -> List[float]:
        """
        Returns a list of floats with length the same as self.frequency_bands. Each float
        is the sum of all convergence magnitudes that occurred at that band.
        :return: List[float]
        """
        # the +1 is necessary since there is a bag above last threshold
        frequency_bands = self.frequency_bands.split(', ')
        magnitudes_bagged = [0] * (len(frequency_bands) + 1)
        frequency_data = self.frequency_data()
        convergence_magnitudes = self.convergence_magnitudes()
        for i, mag in enumerate(convergence_magnitudes):
            if mag != 0:
                frequency = frequency_data[i]
                for j, band in enumerate(frequency_bands):
                    if frequency < band:
                        magnitudes_bagged[j] += mag
                        break
                else:
                    magnitudes_bagged[-1] += mag
        assert (len(magnitudes_bagged) == len(frequency_bands) + 1)
        return magnitudes_bagged

    @clean_df
    @measure
    def divergence_magnitudes_bagged_by_phase(self) -> List[float]:
        """
        Returns a list of floats with length the same as self.phase_bands + 1. Each float
        is the sum of all divergence magnitudes that occurred at that band.
        :return: List[float]
        """
        # the +1 is necessary since there is a bag above last threshold
        phase_bands = self.phase_bands.split(', ')
        magnitudes_bagged = [0] * (len(phase_bands) + 1)
        phase_data = self.phase_data()
        divergence_magnitudes = self.divergence_magnitudes()
        for i, mag in enumerate(divergence_magnitudes):
            if mag != 0:
                phase = phase_data[i]
                for j, band in enumerate(phase_bands):
                    if phase < band:
                        magnitudes_bagged[j] += mag
                        break
                else:
                    magnitudes_bagged[-1] += mag
        assert (len(magnitudes_bagged) == len(phase_bands) + 1)
        return magnitudes_bagged

    @clean_df
    @measure
    def divergence_magnitudes_bagged_by_frequency(self) -> List[float]:
        """
        Returns a list of floats with length the same as self.frequency_bands + 1. Each float
        is the sum of all divergence magnitudes that occurred at that band.
        :return: List[float]
        """
        # the +1 is necessary since there is a bag above last threshold
        frequency_bands = self.frequency_bands.split(', ')
        magnitudes_bagged = [0] * (len(frequency_bands) + 1)
        frequency_data = self.frequency_data()
        divergence_magnitudes = self.divergence_magnitudes()
        for i, mag in enumerate(divergence_magnitudes):
            if mag != 0:
                frequency = frequency_data[i]
                for j, band in enumerate(frequency_bands):
                    if frequency < band:
                        magnitudes_bagged[j] += mag
                        break
                else:
                    magnitudes_bagged[-1] += mag
        assert (len(magnitudes_bagged) == len(frequency_bands) + 1)
        return magnitudes_bagged
