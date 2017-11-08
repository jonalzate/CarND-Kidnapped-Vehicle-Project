/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// random generator
//static default_random_engine rand_gen;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	//====================
	// Set number of particles
	num_particles = 101;

  // Normal dist for adding Gaussian noise to sensor
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  // Initialize all particles
  for (int i = 0; i < num_particles; i++) {
  	// Set values for each particle
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // add Gaussian noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);

		// insert to particles vector
    particles.push_back(p);
  }

	// Set init flag
  is_initialized = true;
	//====================

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	//====================================
	// Normal dist for adding Gaussian noise to sensor
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // calculate new particle state
    // yaw_rate <= zero case
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else { // yaw_rate != zero case
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add Gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
	//====================================


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	//==================================

	// calculate the distance for each observed point to a predicted point
	// iterate through each observation and prediction to calculate shortest distance
	for (unsigned int i = 0; i < observations.size(); i++) {

    // current observation
    LandmarkObs obs = observations[i];

    // initialize minimum distance
    double min_dist = numeric_limits<double>::max();

    // initialize id for landmark
    int map_id = -1;

    for (unsigned int j = 0; j < predicted.size(); j++) {
      // current prediction
      LandmarkObs pred = predicted[j];

      // calculate distance between current and predicted landmarks
      double current_distance = dist(obs.x, obs.y, pred.x, pred.y);

      // check current distance is less than minimum distance and update min_distance
      if (current_distance < min_dist) {
        min_dist = current_distance;
        map_id = pred.id;
      }
    }

    // update observation id with closest prediction id
    observations[i].id = map_id;
  }

	//==================================

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	//==================================
	// iterate over all particles
  for (int i = 0; i < num_particles; i++) {

    // get each particle x, y and theta
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // vector for landmark predictions within sensor_range
    vector<LandmarkObs> predictions;

    // iterate over landmarks and compare sensor ranges
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get id and x,y coordinates
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      // filter landmarks within sensor range of measurement
      if (fabs(landmark_x - particle_x) <= sensor_range && fabs(landmark_y - particle_y) <= sensor_range) {

        // add prediction to vector
        predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    // Compute transformed observations from car to map coordinates
    vector<LandmarkObs> transformed_obs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double transf_x = cos(particle_theta)*observations[j].x - sin(particle_theta)*observations[j].y + particle_x;
      double transf_y = sin(particle_theta)*observations[j].x + cos(particle_theta)*observations[j].y + particle_y;
      transformed_obs.push_back(LandmarkObs{ observations[j].id, transf_x, transf_y });
    }

    // perform data association between predictions and transformed observations
    dataAssociation(predictions, transformed_obs);

    // update particle weight
    particles[i].weight = 1.0;

		// iterate over transformed observations
    for (unsigned int j = 0; j < transformed_obs.size(); j++) {

      // placeholders for observations and predictions x and y
      double obs_x, obs_y, pred_x, pred_y;
      obs_x = transformed_obs[j].x;
      obs_y = transformed_obs[j].y;

      int obs_id = transformed_obs[j].id;

      // iterate over predictions and get x and y values
      for (unsigned int k = 0; k < predictions.size(); k++) {

      	// check that prediction and observation id match
        if (predictions[k].id == obs_id) {

        	// update predicted x and y
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
        }
      }

      // calculate weights for prediction using Multivariate Gaussian Dist
      double landmark_std_x = std_landmark[0];
      double landmark_std_y = std_landmark[1];
      double obs_weight = ( 1/(2*M_PI*landmark_std_x*landmark_std_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(landmark_std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(landmark_std_y, 2))) ) );

      // // update weight with product of current observation weight with total observations weight
      particles[i].weight *= obs_weight;
    }
  }
	//==================================
}

void ParticleFilter::resample() {

	//====================================
	// New vector of particles
	vector<Particle> new_particles;

  // get all current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // Max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // Uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // iterate over random dist samples
  for (int i = 0; i < num_particles; i++) {

    beta += unirealdist(gen) * 2.0;

    while (beta > weights[index]) {

      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
	//====================================

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
