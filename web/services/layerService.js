'use strict';

var _ = require('lodash');
var ol = require('openlayers');
var $ = require('jquery')


angular
    .module('app')
    .factory('LayerService', LayerService);
    
function LayerService($rootScope, $cookies) {
    var service = {};

    service.removeLayer = function(map, layername){
        var layers = [];
        map.getLayers().forEach(function(layer) {
            if(layer.get('name') == layername){
                layers.push(layer);
            }
        })
        console.log('remove layer', layers.length, layername)
        for(var i=0; i<layers.length; i++) map.removeLayer(layers[i]);
    }

    return service;
}