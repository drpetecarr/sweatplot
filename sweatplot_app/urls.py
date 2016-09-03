from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /sweatplot/
    url(r'^$', views.patient_list, name='patient_list'),

    # ex: /sweatplot/ben/
    url(r'^(?P<patient_name>[^/]*)/$', views.patient_view, name='patient_name'),

    # ex: /sweatplot/ben/1
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/$', views.session_view, name='session_number'),

    # ex: /sweatplot/ben/1/graph
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/graph/$', views.graph_view, name='graph'),

    # ex: /sweatplot/ben/1/table
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/table/$', views.table_view, name='table'),

    # ex: /sweatplot/ben/1/convergence_score
    url(r'^(?P<patient_name>.*)/(?P<session_number>[0-9]+)/(?P<measure>.*)/$', views.measure_view, name='measure')
]
