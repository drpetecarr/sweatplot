from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, get_object_or_404
from .models import Patient, Session


def patient_list(request: HttpRequest) -> HttpResponse:
    p_list = [p.name for p in Patient.objects.order_by('-name')[:]]
    context = {'patient_list': p_list}
    return render(request, 'sweatplot_app/patient_list.html', context)


def patient_view(request: HttpRequest, patient_name: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    return render(request, 'sweatplot_app/patient_view.html', {'patient': patient})


def session_view(request: HttpRequest, patient_name: str, session_number: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    return render(request, 'sweatplot_app/session_view.html', {'session': session})


def graph_view(request: HttpRequest, patient_name: str, session_number: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    return render(request, 'sweatplot_app/graph.html', {'session': session})


def table_view(request: HttpRequest, patient_name: str, session_number: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    return render(request, 'sweatplot_app/table.html', {'session': session})


def measure_view(request: HttpRequest, patient_name: str, session_number: str, measure: str) -> HttpResponse:
    patient = get_object_or_404(Patient, name=patient_name)
    session = get_object_or_404(Session, patient=patient, number=session_number)
    result = getattr(session, measure)()
    return render(request, 'sweatplot_app/measure.html', {'result': result})
