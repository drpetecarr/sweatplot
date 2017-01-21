from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, get_object_or_404, get_list_or_404
from .models import Patient, Session, MultipleSessions, create_graph, create_bar_chart, to_csv, \
    create_multiple_bar_charts, multiple_to_csv, create_all_sessions_graph


def patient_list(request: HttpRequest) -> HttpResponse:
    p_list = [p.name for p in Patient.objects.order_by('-name')[:]]
    context = {'patient_list': p_list}
    return render(request, 'sweatplot_app/patient_list.html', context)


def patient_view(request: HttpRequest, patient_name: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    sessions = get_list_or_404(Session, patient=patient)
    multiple_sessions = MultipleSessions(sessions)

    number_of_sessions = min(len(sessions), 999999)
    mean_left_shift = round(sum([s.left_shift() for s in sessions]) / number_of_sessions, 3)
    mean_right_shift = round(sum([s.right_shift() for s in sessions]) / number_of_sessions, 3)
    mean_total_shift = round(sum([s.total_shift() for s in sessions]) / number_of_sessions, 3)
    mean_convergence_score = round(sum([s.convergence_score() for s in sessions]) / number_of_sessions, 3)
    mean_divergence_score = round(sum([s.divergence_score() for s in sessions]) / number_of_sessions, 3)
    mean_mean_convergence_magnitudes = round(
        sum([s.mean_convergence_magnitudes() for s in sessions]) / number_of_sessions, 3)
    mean_mean_divergence_magnitudes = round(
        sum([s.mean_divergence_magnitudes() for s in sessions]) / number_of_sessions, 3)
    mean_median_convergence_magnitudes = round(
        sum([s.median_convergence_magnitudes() for s in sessions]) / number_of_sessions, 3)
    mean_median_divergence_magnitudes = round(
        sum([s.median_divergence_magnitudes() for s in sessions]) / number_of_sessions, 3)
    mean_variance_convergence_magnitudes = round(
        sum([s.variance_convergence_magnitudes() for s in sessions]) / number_of_sessions, 3)
    mean_variance_divergence_magnitudes = round(
        sum([s.variance_divergence_magnitudes() for s in sessions]) / number_of_sessions, 3)
    graph_convergence_magnitudes_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                    '/convergence_magnitudes_against_time/graph'
    graph_divergence_magnitudes_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                   '/divergence_magnitudes_against_time/graph'
    graph_convergence_magnitudes_by_hand_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                            '/convergence_magnitudes_by_hand_against_time/graph'
    graph_divergence_magnitudes_by_hand_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                           '/divergence_magnitudes_by_hand_against_time/graph'
    save_convergence_magnitudes_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                   '/convergence_magnitudes_against_time/to_csv'
    save_divergence_magnitudes_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                  '/divergence_magnitudes_against_time/to_csv'
    save_convergence_magnitudes_by_hand_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                           '/convergence_magnitudes_by_hand_against_time/to_csv'
    save_divergence_magnitudes_by_hand_against_time_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                                          '/divergence_magnitudes_by_hand_against_time/to_csv'
    bar_convergence_by_phase_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                   '/convergence_magnitudes_bagged_by_phases/bar_chart'
    bar_divergence_by_phase_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                  '/divergence_magnitudes_bagged_by_phases/bar_chart'
    bar_convergence_by_frequency_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                       '/convergence_magnitudes_bagged_by_frequencies/bar_chart'
    bar_divergence_by_frequency_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                      '/divergence_magnitudes_bagged_by_frequencies/bar_chart'
    save_convergence_by_phase_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                    '/convergence_magnitudes_bagged_by_phases/to_csv'
    save_divergence_by_phase_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                   '/divergence_magnitudes_bagged_by_phases/to_csv'
    save_convergence_by_frequency_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                        '/convergence_magnitudes_bagged_by_frequencies/to_csv'
    save_divergence_by_frequency_url = '/sweatplot/' + patient.name + '/all_sessions' + \
                                       '/divergence_magnitudes_bagged_by_frequencies/to_csv'

    # frequency_bands_count and phase_bands_counts are lists of lists. Each inner list has 2 objects, the second is
    # a unique bands list, the first is a count of every session with those bands
    frequency_bands_count = []
    for fb in multiple_sessions.frequency_bands_by_session_number():
        frequency_bands_count.append([len(fb[1]), fb[0]])
    phase_bands_count = []
    for pb in multiple_sessions.phase_bands_by_session_number():
        phase_bands_count.append([len(pb[1]), pb[0]])

    return render(request, 'sweatplot_app/patient_view.html',
                  {'patient': patient,
                   'sessions': sessions,
                   'multiple_sessions': multiple_sessions,
                   'number_of_sessions': number_of_sessions,
                   'mean_left_shift': mean_left_shift,
                   'mean_right_shift': mean_right_shift,
                   'mean_total_shift': mean_total_shift,
                   'mean_convergence_score': mean_convergence_score,
                   'mean_divergence_score': mean_divergence_score,
                   'frequency_bands_count': frequency_bands_count,
                   'phase_bands_count': phase_bands_count,
                   'mean_mean_convergence_magnitudes': mean_mean_convergence_magnitudes,
                   'mean_median_convergence_magnitudes': mean_median_convergence_magnitudes,
                   'mean_mean_divergence_magnitudes': mean_mean_divergence_magnitudes,
                   'mean_median_divergence_magnitudes': mean_median_divergence_magnitudes,
                   'mean_variance_convergence_magnitudes': mean_variance_convergence_magnitudes,
                   'mean_variance_divergence_magnitudes': mean_variance_divergence_magnitudes,
                   'graph_convergence_magnitudes_against_time_url': graph_convergence_magnitudes_against_time_url,
                   'graph_divergence_magnitudes_against_time_url': graph_divergence_magnitudes_against_time_url,
                   'graph_convergence_magnitudes_by_hand_against_time_url':
                       graph_convergence_magnitudes_by_hand_against_time_url,
                   'graph_divergence_magnitudes_by_hand_against_time_url':
                       graph_divergence_magnitudes_by_hand_against_time_url,
                   'save_convergence_magnitudes_against_time_url': save_convergence_magnitudes_against_time_url,
                   'save_convergence_magnitudes_by_hand_against_time_url':
                       save_convergence_magnitudes_by_hand_against_time_url,
                   'save_divergence_magnitudes_against_time_url': save_divergence_magnitudes_against_time_url,
                   'save_divergence_magnitudes_by_hand_against_time_url':
                       save_divergence_magnitudes_by_hand_against_time_url,
                   'bar_convergence_by_phase_url': bar_convergence_by_phase_url,
                   'bar_divergence_by_phase_url': bar_divergence_by_phase_url,
                   'bar_convergence_by_frequency_url': bar_convergence_by_frequency_url,
                   'bar_divergence_by_frequency_url': bar_divergence_by_frequency_url,
                   'save_convergence_by_phase_url': save_convergence_by_phase_url,
                   'save_divergence_by_phase_url': save_divergence_by_phase_url,
                   'save_convergence_by_frequency_url': save_convergence_by_frequency_url,
                   'save_divergence_by_frequency_url': save_divergence_by_frequency_url
                   })


def session_view(request: HttpRequest, patient_name: str, session_number: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    graph_convergence_magnitudes = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                   '/convergence_magnitudes/graph'
    save_convergence_magnitudes = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                  '/convergence_magnitudes/to_csv'
    graph_convergence_magnitudes_by_hand = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                           '/convergence_magnitudes_split/graph'
    save_convergence_magnitudes_by_hand = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                          '/convergence_magnitudes_split/to_csv'
    graph_divergence_magnitudes = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                  '/divergence_magnitudes/graph'
    save_divergence_magnitudes = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                 '/divergence_magnitudes/to_csv'
    graph_divergence_magnitudes_by_hand = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                          '/divergence_magnitudes_split/graph'
    save_divergence_magnitudes_by_hand = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                         '/divergence_magnitudes_split/to_csv'
    phase_raw_data = '/sweatplot/' + patient.name + '/' + str(session.number) + '/phase_data'
    convergence_by_phase_bar_chart = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                     '/convergence_magnitudes_bagged_by_phase/bar_chart'
    convergence_by_phase_csv = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                               '/convergence_magnitudes_bagged_by_phase/to_csv'
    divergence_by_phase_bar_chart = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                    '/divergence_magnitudes_bagged_by_phase/bar_chart'
    divergence_by_phase_csv = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                              '/divergence_magnitudes_bagged_by_phase/to_csv'
    frequency_raw_data = '/sweatplot/' + patient.name + '/' + str(session.number) + '/frequency_data'
    convergence_by_frequency_bar_chart = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                         '/convergence_magnitudes_bagged_by_frequency/bar_chart'
    convergence_by_frequency_csv = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                   '/convergence_magnitudes_bagged_by_frequency/to_csv'
    divergence_by_frequency_bar_chart = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                        '/divergence_magnitudes_bagged_by_frequency/bar_chart'
    divergence_by_frequency_csv = '/sweatplot/' + patient.name + '/' + str(session.number) + \
                                  '/divergence_magnitudes_bagged_by_frequency/to_csv'
    return render(request, 'sweatplot_app/session_view.html',
                  {'session': session, 'patient': patient,
                   'graph_convergence_magnitudes': graph_convergence_magnitudes,
                   'save_convergence_magnitudes': save_convergence_magnitudes,
                   'graph_convergence_magnitudes_by_hand': graph_convergence_magnitudes_by_hand,
                   'save_convergence_magnitudes_by_hand': save_convergence_magnitudes_by_hand,
                   'graph_divergence_magnitudes': graph_divergence_magnitudes,
                   'save_divergence_magnitudes': save_divergence_magnitudes,
                   'graph_divergence_magnitudes_by_hand': graph_divergence_magnitudes_by_hand,
                   'save_divergence_magnitudes_by_hand': save_divergence_magnitudes_by_hand,
                   'phase_raw_data': phase_raw_data,
                   'convergence_by_phase_bar_chart': convergence_by_phase_bar_chart,
                   'convergence_by_phase_csv': convergence_by_phase_csv,
                   'divergence_by_phase_bar_chart': divergence_by_phase_bar_chart,
                   'divergence_by_phase_csv': divergence_by_phase_csv,
                   'frequency_raw_data': frequency_raw_data,
                   'convergence_by_frequency_bar_chart': convergence_by_frequency_bar_chart,
                   'convergence_by_frequency_csv': convergence_by_frequency_csv,
                   'divergence_by_frequency_bar_chart': divergence_by_frequency_bar_chart,
                   'divergence_by_frequency_csv': divergence_by_frequency_csv
                   })


def graph_view(request: HttpRequest, patient_name: str, session_number: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    create_graph(getattr(session, measure)(), patient, session)
    return render(request, 'sweatplot_app/graph.html', {})


def csv_view(request: HttpRequest, patient_name: str, session_number: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    to_csv(getattr(session, measure)(), patient_name, measure)
    return render(request, 'sweatplot_app/csv.html', {'patient_name': patient_name, 'session_number': session_number,
                                                      'measure_name': measure})


def all_sessions_csv_view(request: HttpRequest, patient_name: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    sessions = MultipleSessions(get_list_or_404(Session, patient=patient))
    results = getattr(sessions, measure)()
    multiple_to_csv(results, patient_name, measure)
    return render(request, 'sweatplot_app/multiple_csv.html', {'patient_name': patient_name, 'measure_name': measure})


def bar_chart_view(request: HttpRequest, patient_name: str, session_number: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    result = [round(r, 3) for r in getattr(session, measure)()]
    if 'phase' in measure:
        x = [str(z) for z in zip(session.phase_labels.split(', '), ['%.4f' % round(r, 4) for r in result])]
        # create_bar_chart(result, range(len(session.phase_bands.split(', ')) + 1), patient, session)
    elif 'frequency' in measure:
        x = [str(z) for z in zip(session.frequency_labels.split(', '), ['%.4f' % round(r, 4) for r in result])]
        # create_bar_chart(result, range(len(session.frequency_bands.split(', ')) + 1), patient, session)
    else:
        x = [str(z) for z in
             zip(list(range(len(getattr(session, measure)()) + 1)), ['%.4f' % round(r, 4) for r in result])]
        # create_bar_chart(result, range(len(getattr(session, measure)())), patient, session)
    return render(request, 'sweatplot_app/bar_chart.html',
                  {'x': x, 'y': result,
                   'title': patient.name + '  (session: ' + str(session.number) + ')'})


def measure_view(request: HttpRequest, patient_name: str, session_number: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    result = getattr(session, measure)()
    return render(request, 'sweatplot_app/measure.html', {'result': result})


def all_sessions_graph_view(request: HttpRequest, patient_name: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    sessions = MultipleSessions(get_list_or_404(Session, patient=patient))
    create_all_sessions_graph(getattr(sessions, measure)(), patient)
    return render(request, 'sweatplot_app/graph.html', {})


def all_sessions_bar_chart_view(request: HttpRequest, patient_name: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    sessions = MultipleSessions(get_list_or_404(Session, patient=patient))
    results = getattr(sessions, measure)()
    y_data = [df['result'] for df in results]
    lbl_data = [df['bands'] for df in results]
    create_multiple_bar_charts(results, ['temp_plot' + str(i) + '.jpg' for i in range(len(results))], patient)
    return render(request, 'sweatplot_app/multiple_bar_charts.html', {'y': y_data, 'lbls': lbl_data,
                                                                      'sessions': sessions,
                                                                      'patient': patient})
