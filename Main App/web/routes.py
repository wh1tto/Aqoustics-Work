from flask import Flask, request, render_template, redirect, url_for, session, Blueprint, current_app as app
import os
import pandas as pd
import soundfile as sf
from maad import sound
from maad.util import power2dB, plot2d, overlay_rois, format_features
from maad.rois import create_mask, select_rois
from maad.features import shape_features, centroid_features
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import json
from datetime import timedelta

web_bp = Blueprint('web', __name__, template_folder='templates', static_folder='static')

hopespotLinks = [
    {'Abrolhos Bank': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A159'},
    {'Aeolian Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A284'},
    {'Agulhas Front': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A289'},
    {'Alboran Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A249'},
    {'Algoa Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A166'},
    {'Aliwal Shoal': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A167'},
    {'Andaman Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A168'},
    {'Argyll Coast and Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A158'},
    {'Ascension Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A169'},
    {'Atlantis Bank': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A170'},
    {'Azores Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A300'},
    {'Bahamian Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A171'},
    {'Balearic Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A172'},
    {'Bering Sea Deep Canyons': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A173'},
    {'Biological Marine Corridor of Osa': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A322'},
    {'Blue Cavern State Marine Conservation Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A352'},
    {'Blue Shark Central': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A174'},
    {'Bocas del Toro Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A266'},
    {'Bunaken Marine Park': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A271'},
    {'Byron Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A343'},
    {'Cagarras Islands and Surrounding Waters': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A287'},
    {'California Seamounts': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A175'},
    {'Canyon of Caprera': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A350'},
    {'Cape Whale Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A176'},
    {'Capurgana and Cabo Tiburon': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A254'},
    {'Cashes Ledge': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A177'},
    {'Central Arctic Ocean': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A179'},
    {'Chagos Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A180'},
    {'Charlie-Gibbs Fracture Zone': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A181'},
    {'Chichiriviche de la Costa': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A182'},
    {'Chilean Fjords and Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A183'},
    {'Choroni and Chuao': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A184'},
    {'Chumash Heritage National Marine Sanctuary': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A351'},
    {'Coastal Southeast Florida': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A185'},
    {'Coastal Waters of the Black River District': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A162'},
    {'Cocos Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A280'},
    {'Cocos-Galápagos Swimway': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A267'},
    {'Coiba and Cordillera de Coiba': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A281'},
    {'Conflict Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A269'},
    {'Coral Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A292'},
    {'Coral Seamount': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A187'},
    {'Coral Triangle': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A188'},
    {'Costa Rica Thermal Dome': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A178'},
    {'Datan Algal Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A255'},
    {'East Antarctic': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A190'},
    {'East Portland Fish Sanctuary': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A191'},
    {'Eastern Tropical Pacific Seascape': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A192'},
    {'Egg Island, Bahamas': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A193'},
    {'Emperor Seamount Chain': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A194'},
    {'Exmouth Gulf and Ningaloo Coast World Heritage Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A164'},
    {'False Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A195'},
    {'Farm Pond': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A196'},
    {'Fish Rock': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A309'},
    {'Florida Gulf Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A160'},
    {'French Overseas Territories (Wallis and Futuna)': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A197'},
    {'Gakkel Ridge': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A198'},
    {'Galápagos Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A279'},
    {'George Town': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A200'},
    {'Georgia Continental Shelf and Blake Plateau': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A263'},
    {'Ghizilagaj Reserve National Park and Marine Protected Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A260'},
    {'Gold Coast Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A278'},
    {'Golfo Dulce': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A256'},
    {'Gotland': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A277'},
    {'Grand Recif de Toliara': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A201'},
    {'Great Barrier Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A288'},
    {'Great Lakes': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A345'},
    {'Greater Farallones': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A205'},
    {'Greater Skellig Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A342'},
    {'Guanahacabibes National Park': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A250'},
    {'Gulf of California': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A202'},
    {'Gulf of Guinea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A203'},
    {'Gulf of Mexico Deep Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A204'},
    {'Hatteras': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A206'},
    {'Hecate Strait and Queen Charlotte Sound Glass Sponge Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A207'},
    {'Henoko-Ōura Coastal Waters': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A262'},
    {'Hong Kong South': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A334'},
    {'Houtman Abrolhos': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A208'},
    {'Humboldt Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A247'},
    {'Inhambane Seascape': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A326'},
    {'Jaeren Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A248'},
    {'Jangamo Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A330'},
    {'Jardines de la Reina': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A199'},
    {'Kahalu’u Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A328'},
    {'Kangaroo Island North Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A283'},
    {'Kep Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A257'},
    {'Kermadec Trench': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A209'},
    {'Kimbe Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A261'},
    {'Knysna Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A210'},
    {'Kosterfjorden Yttre Hvaler': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A211'},
    {'Laamu Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A324'},
    {'Lakshadweep Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A212'},
    {'Lesvos': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A290'},
    {'Little Cayman': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A275'},
    {'Long Island Marine Management Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A252'},
    {'Lord Howe Rise': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A291'},
    {'Mako Shark Metropolis': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A213'},
    {'Maldive Atolls': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A214'},
    {'Malpelo Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A215'},
    {'Maputaland': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A355'},
    {'Mesoamerican Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A217'},
    {'Misool Marine Reserve': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A219'},
    {'Mohéli': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A282'},
    {'Monterey Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A220'},
    {'Moreton Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A221'},
    {'Myeik Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A216'},
    {'New York-New Jersey Harbor Estuary': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A341'},
    {'New Zealand Coastal Waters': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A270'},
    {'Northeast Iceland': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A344'},
    {'Northwest Passage': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A222'},
    {'Nusa Penida Marine Protected Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A268'},
    {'Ocean Cay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A337'},
    {'Olowalu Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A245'},
    {'Ombai-Wetar Strait': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A276'},
    {'Outer Islands of Seychelles': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A223'},
    {'Pacific Subtropical Convergence Zone': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A224'},
    {'Palau': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A218'},
    {'Palmahim Slide': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A336'},
    {'Palmyra Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A273'},
    {'Pangatalan Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A165'},
    {'Patagonian Shelf': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A225'},
    {'Pearl Islands Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A346'},
    {'Plettenberg Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A226'},
    {'Prince William Sound': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A348'},
    {'Quirimbas Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A227'},
    {'Revillagigedo Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A246'},
    {'Ross Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A228'},
    {'Saba and the Saba Bank': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A338'},
    {'Saint Barthélemy': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A331'},
    {'Saint Vincent and the Grenadines': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A229'},
    {'Salas y Gomez and Nazca Ridges': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A230'},
    {'Salisbury Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A163'},
    {'Salish Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A285'},
    {'San Francisco Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A253'},
    {'Sargasso Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A319'},
    {'Scott Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A232'},
    {'Shinnecock Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A333'},
    {'South San Jorge Gulf': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A340'},
    {'Southeast Shoal of the Grand Banks': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A233'},
    {'St Helena Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A347'},
    {'Subantarctic Islands and Surrounding Seas': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A235'},
    {'Svalbard Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A234'},
    {'Sydney Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A258'},
    {'Tasman Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A236'},
    {'Tavarua Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A274'},
    {'Tenerife-La Gomera': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A265'},
    {'Tetiaroa Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A259'},
    {'The Great Fringing Reef of the Red Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A339'},
    {'The Great Southern Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A264'},
    {'Tribuga Gulf': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A161'},
    {'Tropical Pacific Sea of Peru': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A238'},
    {'Tubbataha Reefs Natural Park': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A329'},
    {'Varadero’s Coral Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A272'},
    {'Verde Island Passage': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A349'},
    {'Walter’s Shoal': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A240'},
    {'Western Pacific Donut Hole 1': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A241'},
    {'Western Pacific Donut Hole 2': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A242'},
    {'Western Pacific Donut Hole 3': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A243'},
    {'Western Pacific Donut Hole 4': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A244'},
    {'Whale and Dolphin Sanctuary of Uruguay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A251'},
    {'White Shark Cafe': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A237'},
    {'Wider Vatika Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A239'},
    {'Wotho Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A312'}
]

@web_bp.route('/')
def home():
    return render_template('index.html')

@web_bp.route('/upload', methods=['POST'])
def upload_file():
    location = request.form.get('location')
    filename = request.form.get('filename')
    
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        all_timestamps = []
        
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = '.wav'  # Default to .wav if no extension provided
        
        camel_case_location = to_camel_case(location)
        full_filename = f"{filename}{file_extension}"
        
        # Create directory path using the filename
        hopespot_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', camel_case_location)
        audio_dir = os.path.join(hopespot_dir, 'audio', filename)
        os.makedirs(audio_dir, exist_ok=True)
        
        # Initialize audio_data.json if it does not exist
        audio_data_path = os.path.join(hopespot_dir, 'audio_data.json')
        if not os.path.exists(audio_data_path):
            with open(audio_data_path, 'w') as f:
                json.dump({}, f)

        file_path = os.path.join(audio_dir, full_filename)
        file.save(file_path)
        timestamps = process_audio(file_path)
        all_timestamps.append(timestamps)

        # Load or initialize audio_data.json from the hopespot folder
        if os.path.exists(audio_data_path):
            try:
                with open(audio_data_path, 'r') as f:
                    audio_data = json.load(f)
            except json.JSONDecodeError:
                # Handle empty or corrupted JSON file
                audio_data = {}
        else:
            audio_data = {}

        # Convert timestamps to the specified format
        formatted_timestamps = [
            {"start": format_timestamp(row['begin']), "end": format_timestamp(row['end']), "note": "note", "score": "3"}
            for _, row in timestamps.iterrows()
        ]

        # Update the JSON structure
        if full_filename not in audio_data:
            audio_data[full_filename] = {"timestamps": formatted_timestamps, "votes": 0}
        else:
            audio_data[full_filename]["timestamps"].extend(formatted_timestamps)

        # Write back to the audio_data.json file
        with open(audio_data_path, 'w') as f:
            json.dump(audio_data, f, indent=4)

        
        session['all_timestamps'] = [df.to_dict(orient='records') for df in all_timestamps]
        session['titles'] = timestamps.columns.values.tolist()
        session['location'] = location
        session['filename'] = filename
        
        return redirect(url_for('web.result'))

@web_bp.route('/result')
def result():
    all_timestamps = [pd.DataFrame(data) for data in session.get('all_timestamps', [])]
    titles = session.get('titles', [])
    location = session.get('location', '')
    filename = session.get('filename', '')
    
    locationLink = searchDict(hopespotLinks, location)
    camel_case_location = to_camel_case(location)
    audio_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', camel_case_location, 'audio', filename)

    spectrograms = []
    
    # Create a spectogram for each new clip created, appending the same number as in file name (clip_0.wav, clip_1.wav, etc.)
    # Save the plot to the same folder as the clip
    # TODO: Do we want spectogram to permanently save to the folder or just display on the page? (storage vs processing time/overhead)
    # TODO: Add new results page for individual recordings instead of using the results page, change references and add voting functionality
    # TODO: Also add voting functionality for indiivudal clips 
    
    for file in os.listdir(audio_folder):
        if file.startswith('clip') and file.endswith('.wav'):
            clip_path = os.path.join(audio_folder, file)
            plot_path = os.path.join(audio_folder, f"{os.path.splitext(file)[0]}_plot.png")
            plotstft(clip_path, plot_path)
            relative_plot_path = f'hopespots/{camel_case_location}/audio/{filename}/{os.path.basename(plot_path)}'
            spectrograms.append(relative_plot_path)
    
    
    return render_template('result.html', tables=all_timestamps, titles=titles, location=location, locationLink=locationLink, spectrograms=spectrograms)

# Route to list all hopespots
@web_bp.route('/hopespots')
def hopespots():
    hopespots = []
    for h in hopespotLinks:
        name = list(h.keys())[0]
        clips_count = len(get_audio_files_hopespots(name))
        hopespots.append({'name': name, 'clips_count': clips_count})
    
    hopespots.sort(key=lambda x: x['name'])  # Initially sort by name
    
    return render_template('hopespots.html', hopespots=hopespots)

# Route to display details for a specific hopespot
@web_bp.route('/hopespot/<hopespot_name>')
def hopespot(hopespot_name):
    
    hopespots = [h for h in hopespotLinks if hopespot_name in h]
    if not hopespots:
        return "Hopespot not found", 404

    hopespot_link = hopespots[0].get(hopespot_name)
    audio_files = get_audio_files(hopespot_name)
    
    return render_template('hopespot.html', hopespot_name=hopespot_name, link=hopespot_link, audio_files=audio_files)

# Route to display details for a specific audio recording
@web_bp.route('/hopespot/<hopespot_name>/audio/<audio_filename>')
def audio(hopespot_name, audio_filename):
    hopespot_name_no_cc = hopespot_name
    hopespot_name = to_camel_case(hopespot_name)
    audio_data = get_audio_data(hopespot_name, audio_filename)
    
    if not audio_data:
        return "Audio file not found", 404

    num_clips = len(audio_data['timestamps'])
    
    audio_filename_no_ext = os.path.splitext(audio_filename)[0]
    
    # Run loop over number of clips to retrieve the clip_0.wav, clip_1.wav, etc
    # Also add the path for the spectogram for each clip (instead of .wav, use _plot.png)
    
    # FIXME: Fix spectograms (wont always be in order 0-1-2-3-4, some may be 1-3-8-15-20)
    clips = []
    spectograms = []
    for i in range(num_clips):
        clip_filename = f"clip_{i}.wav"
        clip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', hopespot_name, 'audio', audio_filename_no_ext, clip_filename)
        spectogram_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', hopespot_name, 'audio', audio_filename_no_ext, f"clip_{i}_plot.png")
        clips.append(clip_path)
        spectograms.append(spectogram_path)
    
    return render_template('audio.html', hopespot_name=hopespot_name, audio_filename=audio_filename, audio_data=audio_data, audio_filename_no_ext=audio_filename_no_ext, clips=clips, spectograms=spectograms, hopespot_name_no_cc=hopespot_name_no_cc)

# Utility functions
def get_audio_files(hopespot_name):
    camel_case_hopespot_name = to_camel_case(hopespot_name)
    base_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', camel_case_hopespot_name, 'audio')
    audio_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', camel_case_hopespot_name, 'audio_data.json')
    
    if not os.path.exists(base_path) or not os.path.exists(audio_data_path):
        return []
    
    audio_files = []
    with open(audio_data_path, 'r') as f:
        audio_data = json.load(f)

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            filename = folder + '.WAV'
            if filename in audio_data:
                votes = audio_data[filename].get('votes', 0)
                audio_files.append({'filename': filename, 'votes': votes})
            elif folder + '.wav' in audio_data:
                votes = audio_data[folder + '.wav'].get('votes', 0)
                audio_files.append({'filename': folder + '.wav', 'votes': votes})
    
    audio_files.sort(key=lambda x: x['votes'], reverse=True)
    
    return audio_files

def get_audio_files_hopespots(hopespot_name):
    # Get all files within hopespots
    base_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', to_camel_case(hopespot_name), 'audio')
    if not os.path.exists(base_path):
        return []

    audio_files = []
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            audio_filename = f"{entry}.wav"
            audio_file_path = os.path.join(entry_path, audio_filename)
            if os.path.exists(audio_file_path):
                audio_files.append(audio_filename)

    return audio_files

def get_audio_data(hopespot_name, audio_filename):
    base_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hopespots', hopespot_name, 'audio_data.json')
    if not os.path.exists(base_path):
        return None

    with open(base_path) as json_file:
        audio_data = json.load(json_file)
    
    return audio_data.get(audio_filename)


def process_audio(file_path):
    # Load the audio file
    s, fs = sound.load(file_path)
    s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')

    # Spectrogram parameters
    db_max = 70
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Background removal and smoothing
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=2, bin_per=0.25)
    im_rois, df_rois = select_rois(im_mask, min_roi=50, max_roi=None)

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Calculate shape and centroid features
    df_shape, params = shape_features(Sxx_db, resolution='low', rois=df_rois)
    df_centroid = centroid_features(Sxx_db, df_rois)

    # Get median frequency and normalize
    median_freq = fn[np.round(df_centroid.centroid_y).astype(int)]
    df_centroid['centroid_freq'] = median_freq / fn[-1]

    # t-SNE for dimensionality reduction
    X = df_shape.loc[:, df_shape.columns.str.startswith('shp')]
    X = X.join(df_centroid.centroid_freq)  # add column and normalize values

    tsne = TSNE(n_components=2, perplexity=12, init='pca', verbose=True)
    Y = tsne.fit_transform(X)

    # Clustering using DBSCAN
    cluster = DBSCAN(eps=5, min_samples=4).fit(Y)

    # Overlay bounding box on the original spectrogram
    df_rois['label'] = cluster.labels_.astype(str)
    overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':60, 'extent':ext})

    # Filter ROIs for those with centroid frequency below 1000Hz
    low_freq_rois = df_rois[df_centroid['centroid_freq'] * fn[-1] < 1000]

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps.columns = ['begin', 'end']

    audio_clips = []
    for i, row in low_freq_timestamps.iterrows():
        # Add half a second (0.5 seconds) to the beginning and end of the clip
        start_sample = int(max(0, (row['begin'] - 0.5) * fs))
        end_sample = int(min(len(s), (row['end'] + 0.5) * fs))
        audio_clip = s[start_sample:end_sample]
        clip_filename = f'clip_{i}.wav'
        clip_path = os.path.join(os.path.dirname(file_path), clip_filename)
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append(clip_filename)

    low_freq_timestamps['audio_clip'] = audio_clips

    return low_freq_timestamps

def searchDict(dict, search):
    for item in dict:
        if search in item:
            return item[search]
    return None

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, plotLocation, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    plt.savefig(plotLocation, bbox_inches="tight")
    plt.clf()

    return ims

def to_camel_case(s):
    parts = s.split()
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    return str(td)