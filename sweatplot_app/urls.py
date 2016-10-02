from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /sweatplot/
    url(r'^$', views.patient_list, name='patient_list'),

    # ex: /sweatplot/ben/
    url(r'^(?P<patient_name>[^/]*)/$', views.patient_view, name='patient_name'),

    # ex: /sweatplot/ben/1
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/$', views.session_view, name='session_number'),

    # ex: /sweatplot/ben/1/convergence_magnitudes/graph
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/(?P<measure>.*)/graph/$', views.graph_view, name='graph'),

    # ex: /sweatplot/ben/1/divergence_magnitudes/bar_chart
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/(?P<measure>.*)/bar_chart/$', views.bar_chart_view,
        name='bar_chart'),

    # ex: /sweatplot/ben/1/convergence_score
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/(?P<measure>[^/]*)/$', views.measure_view, name='measure'),

    # ex: /sweatplot/ben/1/convergence_score/to_csv/
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/(?P<measure>.*)/to_csv/$', views.csv_view, name='csv'),

    # ex: /sweatplot/henry/all_sessions/convergence_magnitudes_by_hand_against_time/graph
    url(r'^(?P<patient_name>.*)/all_sessions/(?P<measure>.*)/graph', views.all_sessions_graph_view,
        name='all_sessions_graph'),

    # ex: /sweatplot/henry/all_sessions/convergence_magnitudes_by_hand_against_time/bar_chart
    url(r'^(?P<patient_name>.*)/all_sessions/(?P<measure>.*)/bar_chart', views.all_sessions_bar_chart_view,
        name='all_sessions_bar_chart'),

    # ex: /sweatplot/henry/all_sessions/convergence_magnitudes_by_hand_against_time/to_csv
    url(r'^(?P<patient_name>.*)/all_sessions/(?P<measure>.*)/to_csv', views.all_sessions_csv_view,
        name='all_sessions_csv')
]
