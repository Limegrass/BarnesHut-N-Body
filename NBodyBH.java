/**
 * NBodyBH.java
 *
 * Implementation of the Barnes-Hut algorithm with MPI Parallelism
 * Modified from a base implemention from chindesaurus from github at
 * https://github.com/chindesaurus/BarnesHut-N-Body
 * Added MPI parallelism to the code along with changing the file input
 * to a random mass generation, for project simplification and focusing on the parallelism
 *
 * Compilation:  javac NBodyBH.java
 * Execution:    java NBodyBH < inputs/[filename].txt
 * Dependencies: BHTree.java Body.java Quad.java StdDraw.java
 * Input files:  ./inputs/*.txt
 *
 * @author chindesaurus
 * @author James Ni
 * @version 1.01
 */

import java.awt.Color;
import mpi.Comm;
import mpi.MPI;
import mpi.MPIException;
import java.nio.DoubleBuffer;
import java.util.Random;

public class NBodyBH {

	public static void main(String[] args) throws MPIException{

		//Initialize MPI variables
		MPI.Init(args);
		Comm comm = MPI.COMM_WORLD;
		int rank = comm.getRank();
		int size = comm.getSize();
		System.out.println("My Rank is " + rank	);

		try {
			//Initialization stage
			final double threshold = .5;  //threshold ratio of the size of a cell/distance to the cell's center of mass
			final double dt = 0.1;                     // time quantum
			int N = 4000;								// Total number of masses to generate
			double radius = 2.80000E06;					// Total radius of the size of the space
			int portion = (int)(double)N/size;			//Local portion of the total data

			// turn on animation mode and rescale coordinate system
			StdDraw.show(0);
			StdDraw.setXscale(-radius, +radius);
			StdDraw.setYscale(-radius, +radius);

			// Initialize Randomize Bodies
			Body[] bodies = new Body[portion];
			Random numGen = new Random();
			for (int i = 0; i <portion ; i++) {
				double px   = numGen.nextDouble()*radius;
				double py   = numGen.nextDouble()*radius;
				//Also initialize with small but random velocities
				double vx   = -numGen.nextDouble()*numGen.nextDouble()*numGen.nextDouble()*radius*.1;
				double vy   = -numGen.nextDouble()*numGen.nextDouble()*numGen.nextDouble()*radius*.1;
				if(rank%2==0){
					px*=-1;
					vx*=-1;
				}
				if(rank>(size/2)-1){
					py*=-1;
					vy*=-1;
				}
				double mass = 6.40000E21;
				int red     = 255;
				int green   = 255;
				int blue    = 0;
				bodies[i]   = new Body(px, py, vx, vy, mass, new Color(red, green, blue));
			}
			//I place a body of large mass to easily visualize that my other processes are properly interacting with
			//the one blackhole (very large mass object) located in the first process.
			if(rank==0){
				bodies[0] = new Body(0, 0, 0, 0, 6.40000E26, new Color(255, 0, 0));
			}
			//Additionally, to place a body at the center of the process for alternative testing
			/*
			double px = radius/2;
			double py = radius/2;
			if(rank%2==0){
				px*=-1;
			}
			if(rank>(size/2)-1){
				py*=-1;
			}
			bodies[0] = new Body(px, py, 0, 0 , 6.40000E26, new Color(255, 0, 0));
			*/


			//Simulation stage
			for (double t = 0.0; true; t = t + dt) {

				//Initialize the Quad and Barnes-Hut Tree
				Quad quad = new Quad(0, 0, radius * 2);
				BHTree tree = new BHTree(quad, threshold);

				// build the Barnes-Hut tree
				// Ignores bodies that has gone off the screen
				for(int i = 0; i<portion; i++){
					if (bodies[i].in(quad)){
						tree.insert(bodies[i]);
					}
				}

				//Calculates local contributions to each body in the system.
				for(int i = 0; i< portion; i++){
					bodies[i].resetForce();
					tree.updateForce(bodies[i], 0);
				}


				//Creates buffers for giving x and y position, and mass
				//Reuse the same buffers to receive the calculated force back
				DoubleBuffer xBuffer = MPI.newDoubleBuffer(portion),
						yBuffer = MPI.newDoubleBuffer(portion), massBuffer = MPI.newDoubleBuffer(portion);

				for(int r=1; r<size; r++){
					int toRank = (rank+r)%size;
					int fromRank = (rank-r)%size;
					//Modulo of a negative returns a negative in the version of Java I am using
					// So I account for it by making sure it uses the Mathematical version of modulo size by adding
					// The size
					if(fromRank<0){
						fromRank+=size;
					}
					for(int i=0; i<portion; i++)
					{
						// Fills the buffers with position and mass information of the processor's
						// bodies to send to other processes
						xBuffer.put(i,bodies[i].rx());
						yBuffer.put(i, bodies[i].ry());
						massBuffer.put(i, bodies[i].mass());
					}

					//Sends the mass and position of this body and receive it from a successive body
					comm.sendRecvReplace(massBuffer, portion, MPI.DOUBLE, toRank, 100, fromRank, MPI.ANY_TAG);
					comm.sendRecvReplace(xBuffer, portion, MPI.DOUBLE, toRank, 100, fromRank, MPI.ANY_TAG);
					comm.sendRecvReplace(yBuffer, portion, MPI.DOUBLE, toRank, 100, fromRank, MPI.ANY_TAG);

					//Create a placeholder body with the position and mass given by the buffers
					//and calculate the total force on it by the Barnes-Hut tree
					for(int i=0; i<portion; i++)
					{
						//Receive the incoming requests for force processing with position and mass
						Body temp = new Body(xBuffer.get(i), yBuffer.get(i), 0, 0, massBuffer.get(i), new Color(0, 0, 0));
						tree.updateForce(temp, 0);
						//Replace buffers with x and y force and send them back
						xBuffer.put(i, temp.fx());
						yBuffer.put(i,  temp.fy());
					}

					//Sends the calculated force back to it's original process and receives it from another process
					comm.sendRecvReplace(xBuffer, portion, MPI.DOUBLE, fromRank, 100, toRank, MPI.ANY_TAG);
					comm.sendRecvReplace(yBuffer, portion, MPI.DOUBLE, fromRank, 100, toRank, MPI.ANY_TAG);

					// Use calculated force to update the total force on each body
					for(int i=0; i<portion; i++)
					{
						bodies[i].addForce(xBuffer.get(i), yBuffer.get(i));
					}
				}



				// I am not worried about the inefficiencies in message passing here as it is a trivial part needed
				// only for the visualization
				// The update function needs to happen normally, but the replacement into the buffers does not
				// If visualization was not a concern.
				for(int i=0; i<portion; i++){
					bodies[i].update(dt);
					xBuffer.put(i, bodies[i].rx());
					yBuffer.put(i, bodies[i].ry());
				}

				//For visualization, I choose to allGather so I display the same
				//Simulation on all windows (The visualization opens it for every process)
				//This portion can be completely deleted if we wanted to say
				//Print out the current location of every body and save it
				//If there was more time, designing a better visualization and
				//having each process display their points all onto a single
				//window would be ideal.

				StdDraw.clear(StdDraw.BLACK);
				DoubleBuffer allX = MPI.newDoubleBuffer(N);
				DoubleBuffer allY = MPI.newDoubleBuffer(N);
				comm.allGather(xBuffer, portion, MPI.DOUBLE,allX, portion, MPI.DOUBLE);
				comm.allGather(yBuffer, portion, MPI.DOUBLE, allY, portion, MPI.DOUBLE);


				Body bHole = new Body(allX.get(0), allY.get(0), 0, 0, 0, new Color(255, 0, 0));
				bHole.draw();
				for(int i = 1; i< N; i++)
				{
					Body temp = new Body(allX.get(i), allY.get(i), 0, 0, 0, new Color(255, 255, 0));
					temp.draw();
				}

				comm.barrier(); //Draws bodies on all processors first before displaying it

				StdDraw.show(10);

				//}
			}
		}finally{
			MPI.Finalize();
		}
	}
}
