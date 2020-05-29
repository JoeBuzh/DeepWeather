'use strict';
var _ = require('lodash');
var moment = require('moment');

require('../services/olService');
var ol = require('openlayers');



angular
    .module('app')
    .controller('ObsController', ObsController);


function ObsController(OlService, $rootScope, $cookies, $routeParams, $scope, $http, $filter, $q, $compile) {
    var vm = this;

}
